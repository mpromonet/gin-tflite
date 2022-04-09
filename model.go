package main

import (
	"image"
	"log"
	"time"

	"github.com/mattn/go-tflite"
	"github.com/mattn/go-tflite/delegates/edgetpu"
	"gocv.io/x/gocv"
)

type Model struct {
	model    *tflite.Model
	interp   *tflite.Interpreter
	postproc PostProcessing
}

func NewModel(modelPath string, postproc PostProcessing) *Model {
	model := tflite.NewModelFromFile(modelPath)
	if model == nil {
		log.Println("cannot load model")
		return nil
	}

	options := tflite.NewInterpreterOptions()
	defer options.Delete()

	options.SetNumThread(4)

	// add TPU
	devices, err := edgetpu.DeviceList()
	if err != nil {
		log.Printf("Could not get EdgeTPU devices: %v", err)
	}
	if len(devices) == 0 {
		log.Println("No edge TPU devices found")
	} else {
		options.AddDelegate(edgetpu.New(devices[0]))
	}

	interpreter := tflite.NewInterpreter(model, options)
	if interpreter == nil {
		log.Println("cannot create interpreter")
		return nil
	}

	status := interpreter.AllocateTensors()
	if status != tflite.OK {
		log.Print("allocate failed")
		return nil
	}
	return &Model{model, interpreter, postproc}
}

func argmax(f []float32) (int, float32) {
	r, m := 0, f[0]
	for i, v := range f {
		if v > m {
			m = v
			r = i
		}
	}
	return r, m
}

func getLabel(labels []string, class int) string {
	label := "unknown"
	if class < len(labels) {
		label = labels[class]
	}
	return label
}

func getTensorShape(tensor *tflite.Tensor) []int {
	shape := []int{}
	for idx := 0; idx < tensor.NumDims(); idx++ {
		shape = append(shape, tensor.Dim(idx))
	}
	return shape
}

func fillInput(input *tflite.Tensor, img gocv.Mat) {
	wanted_height := input.Dim(1)
	wanted_width := input.Dim(2)
	resized := gocv.NewMat()
	switch input.Type() {
	case tflite.UInt8:
		img.ConvertTo(&resized, gocv.MatTypeCV32F)
		gocv.Resize(resized, &resized, image.Pt(wanted_width, wanted_height), 0, 0, gocv.InterpolationDefault)
		if v, err := resized.DataPtrFloat32(); err == nil {
			ptr := make([]uint8, len(v))
			for i := 0; i < len(v); i++ {
				ptr[i] = uint8(v[i])
			}
			input.SetUint8s(ptr)
		}
	case tflite.Float32:
		img.ConvertTo(&resized, gocv.MatTypeCV32F)
		gocv.Resize(resized, &resized, image.Pt(wanted_width, wanted_height), 0, 0, gocv.InterpolationDefault)
		if v, err := resized.DataPtrFloat32(); err == nil {
			for i := 0; i < len(v); i++ {
				v[i] = v[i] / 255.5
			}
			input.SetFloat32s(v)
		}
	}
	resized.Close()
}

func filterOutput(bboxes []image.Rectangle, confidences []float32, classes []int, scoreTh float32, nmsTh float32, labels []string) []item {
	var items []item

	if len(bboxes) > 0 {
		indices := make([]int, len(bboxes))
		for i := range indices {
			indices[i] = -1
		}
		gocv.NMSBoxes(bboxes, confidences, scoreTh, nmsTh, indices)

		for _, idx := range indices {
			if idx >= 0 {
				classID := classes[idx]
				confidence := confidences[idx]
				bbox := bboxes[idx]
				item := item{ClassID: classID,
					ClassName: getLabel(labels, classID),
					Score:     confidence,
					Box:       bbox}
				log.Println(item)
				items = append(items, item)
			}
		}
	}
	return items
}

func (m *Model) modelWorker(scoreTh float32, nmsTh float32, labels []string, in chan gocv.Mat, out chan []item) {
	idleDuration := 10 * time.Millisecond
	timeout := time.NewTimer(idleDuration)
	defer timeout.Stop()
	for {
		timeout.Reset(idleDuration)
		select {
		case img := <-in:
			if !img.Empty() {
				// fill input tensor
				input := m.interp.GetInputTensor(0)
				log.Println("input shape:", input.Name(), getTensorShape(input), input.Type(), input.QuantizationParams())
				fillInput(input, img)

				// inference
				status := m.interp.Invoke()
				log.Printf("status: %v", status)
				if status != tflite.OK {
					log.Println("invoke failed")
				}

				// convert output
				bboxes := []image.Rectangle{}
				confidences := []float32{}
				classes := []int{}
				for idx := 0; idx < m.interp.GetOutputTensorCount(); idx++ {
					output := m.interp.GetOutputTensor(idx)
					log.Println("output:", output.Name(), getTensorShape(output), output.Type(), output.QuantizationParams())
					bboxes_, confidences_, classes_ := m.postproc.extractBoxes(output, scoreTh, float32(img.Cols()), float32(img.Rows()))
					bboxes = append(bboxes, bboxes_...)
					confidences = append(confidences, confidences_...)
					classes = append(classes, classes_...)
				}

				// NMS
				items := filterOutput(bboxes, confidences, classes, scoreTh, nmsTh, labels)

				out <- items
				img.Close()
			} else {
				return
			}
		case <-timeout.C:
		}
	}
}
