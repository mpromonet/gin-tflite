package main

import (
	"image"
	"log"
	"math"
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

func extractOutput(output *tflite.Tensor, scoreTh float32, width float32, height float32) ([]image.Rectangle, []float32, []int) {

	var loc []float32
	outputtype := output.Type()
	switch outputtype {
	case tflite.UInt8:
		f := output.UInt8s()
		loc = make([]float32, len(f))
		for i, v := range f {
			loc[i] = float32(v) / 255
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
		if output.NumDims() == 3 {
			totalSize := 1
			for idx := 0; idx < output.NumDims(); idx++ {
				totalSize *= output.Dim(idx)
			}
			for idx := 0; idx < totalSize; idx += 85 {
				if loc[idx+4] > scoreTh {
					x := int(loc[idx+0] * width)
					y := int(loc[idx+1] * height)
					w := int(loc[idx+2] * width)
					h := int(loc[idx+3] * height)
					bboxes = append(bboxes, image.Rect(x-w/2, y-h/2, x+w/2, y+h/2))
					classId, score := argmax(loc[idx+5 : idx+85])
					confidences = append(confidences, score)
					classes = append(classes, classId)
				}
			}
		} else if output.NumDims() == 4 {
			sx := float32(width) / float32(output.Dim(1))
			sy := float32(height) / float32(output.Dim(2))
			for i := 0; i < output.Dim(1); i++ {
				for j := 0; j < output.Dim(2); j++ {
					idx := (i*output.Dim(2) + j) * output.Dim(output.NumDims()-1)
					if loc[idx+4] > scoreTh {
						dx := float32(10.0)
						dy := float32(13.0)
						x := sx*float32(j) + sx*loc[idx+0]
						y := sy*float32(i) + sy*loc[idx+1]
						w := sx * float32(math.Log(float64(dx*float32(math.Exp(float64(loc[idx+2]))))))
						h := sy * float32(math.Log(float64(dy*float32(math.Exp(float64(loc[idx+3]))))))
						bboxes = append(bboxes, image.Rect(int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2)))
						classId, score := argmax(loc[idx+5 : idx+85])
						confidences = append(confidences, score)
						classes = append(classes, classId)
					}
				}
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
