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
	model  *tflite.Model
	interp *tflite.Interpreter
}

func NewModel(modelPath string) *Model {
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
	return &Model{model, interpreter}
}

func fillInput(input *tflite.Tensor, img gocv.Mat) {
	wanted_height := input.Dim(1)
	wanted_width := input.Dim(2)
	resized := gocv.NewMat()
	switch input.Type() {
	case tflite.UInt8:
		qp := input.QuantizationParams()
		img.ConvertTo(&resized, gocv.MatTypeCV32F)
		gocv.Resize(resized, &resized, image.Pt(wanted_width, wanted_height), 0, 0, gocv.InterpolationDefault)
		if v, err := resized.DataPtrFloat32(); err == nil {
			ptr := make([]uint8, len(v))
			for i := 0; i < len(v); i++ {
				ptr[i] = uint8(v[i]/float32(qp.Scale) + float32(qp.ZeroPoint))
			}
			input.SetUint8s(ptr)
		}
	case tflite.Float32:
		img.ConvertTo(&resized, gocv.MatTypeCV32F)
		gocv.Resize(resized, &resized, image.Pt(wanted_width, wanted_height), 0, 0, gocv.InterpolationDefault)
		if v, err := resized.DataPtrFloat32(); err == nil {
			for i := 0; i < len(v); i++ {
				v[i] = (v[i] - 127.5) / 127.5
			}
			input.SetFloat32s(v)
		}
	}
	resized.Close()
}

func extractOutput(output *tflite.Tensor, scoreTh float32, w float32, h float32) ([]image.Rectangle, []float32, []int) {

	var loc []float32
	outputtype := output.Type()
	switch outputtype {
	case tflite.UInt8:
		f := output.UInt8s()
		loc = make([]float32, len(f))
		qp := output.QuantizationParams()
		for i, v := range f {
			loc[i] = (float32(v) - float32(qp.ZeroPoint)) * float32(qp.Scale)
		}
	case tflite.Float32:
		f := output.Float32s()
		loc = make([]float32, len(f))
		for i, v := range f {
			loc[i] = v
		}
	}

	bboxes := []image.Rectangle{}
	confidences := []float32{}
	classes := []int{}
	if len(loc) != 0 {
		totalSize := 1
		for idx := 0; idx < output.NumDims(); idx++ {
			totalSize *= output.Dim(idx)
		}
		for idx := 0; idx < totalSize; idx += 85 {
			if loc[idx+4] > scoreTh {
				x1 := int(loc[idx+0] * w)
				y1 := int(loc[idx+1] * h)
				w := int(loc[idx+2] * w)
				h := int(loc[idx+3] * h)
				bboxes = append(bboxes, image.Rect(x1-w/2, y1-h/2, x1+w/2, y1+h/2))
				classId, score := argmax(loc[idx+5 : idx+85])
				confidences = append(confidences, score)
				classes = append(classes, classId)
			}
		}
	}

	return bboxes, confidences, classes
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
					bboxes_, confidences_, classes_ := extractOutput(output, scoreTh, float32(img.Cols()), float32(img.Rows()))
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
