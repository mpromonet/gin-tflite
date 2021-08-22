package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"io/ioutil"
	"log"
	"net/http"
	"os"

	"github.com/mattn/go-tflite"

	"gocv.io/x/gocv"
)

func loadLabels(filename string) ([]string, error) {
	labels := []string{}
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		labels = append(labels, scanner.Text())
	}
	return labels, nil
}

type item struct {
	Box       image.Rectangle
	Score     float32
	ClassID   int
	ClassName string
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

func createInterpreter(modelPath string) (*tflite.Model, *tflite.Interpreter) {
	model := tflite.NewModelFromFile(modelPath)
	if model == nil {
		log.Println("cannot load model")
		return model, nil
	}

	options := tflite.NewInterpreterOptions()

	options.SetNumThread(4)
	defer options.Delete()

	interpreter := tflite.NewInterpreter(model, options)
	if interpreter == nil {
		log.Println("cannot create interpreter")
		return model, nil
	}

	status := interpreter.AllocateTensors()
	if status != tflite.OK {
		log.Print("allocate failed")
		return model, nil
	}
	return model, interpreter
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
		gocv.Resize(img, &resized, image.Pt(wanted_width, wanted_height), 0, 0, gocv.InterpolationDefault)
		if v, err := resized.DataPtrUint8(); err == nil {
			copy(input.UInt8s(), v)
		}
	case tflite.Float32:
		img.ConvertTo(&resized, gocv.MatTypeCV32F)
		gocv.Resize(resized, &resized, image.Pt(wanted_width, wanted_height), 0, 0, gocv.InterpolationDefault)
		if v, err := resized.DataPtrFloat32(); err == nil {
			for i := 0; i < len(v); i++ {
				v[i] = (v[i] - 127.5) / 127.5
			}
			copy(input.Float32s(), v)
		}
	}
	resized.Close()
}

func extractOutput(output *tflite.Tensor, scoreTh float32, w float32, h float32) ([]image.Rectangle, []float32, []int) {

	var loc []float32
	shape := getTensorShape(output)
	outputtype := output.Type()
	switch outputtype {
	case tflite.UInt8:
		f := output.UInt8s()
		loc = make([]float32, len(f))
		for i, v := range f {
			loc[i] = float32(v)
		}
	case tflite.Float32:
		f := output.Float32s()
		loc = make([]float32, len(f))
		for i, v := range f {
			loc[i] = v
		}
	}
	log.Printf("len: %v", len(loc))

	bboxes := []image.Rectangle{}
	confidences := []float32{}
	classes := []int{}
	if len(loc) != 0 {
		for i := 0; i < shape[1]; i++ {
			idx := (i * shape[2])
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

func main() {
	modelPath := flag.String("model", "models/lite-model_yolo-v5-tflite_tflite_model_1.tflite", "path to model file")
	labelPath := flag.String("label", "models/coco.names", "path to label file")
	scoreTh := flag.Float64("score", 0.3, "score threshold")
	nmsTh := flag.Float64("nms", 0.5, "nms threshold")

	flag.Parse()

	labels, err := loadLabels(*labelPath)
	if err != nil {
		log.Fatal(err)
	}

	model, interpreter := createInterpreter(*modelPath)
	if interpreter == nil {
		log.Println("cannot create interpreter")
		return
	}
	defer model.Delete()
	defer interpreter.Delete()

	// start http server
	http.Handle("/", http.FileServer(http.Dir("./static")))
	http.HandleFunc("/runmodel", func(w http.ResponseWriter, r *http.Request) {
		body := r.Body
		fileBytes, err := ioutil.ReadAll(body)
		if err != nil {
			fmt.Println(err)
		}
		log.Println("Header:", r.Header, "Body Size: ", len(fileBytes))
		body.Close()

		// decode image
		img, err := gocv.IMDecode(fileBytes, gocv.IMReadUnchanged)
		if err != nil {
			log.Println(err)
		}

		// fill input tensor
		input := interpreter.GetInputTensor(0)
		log.Println("input shape:", input.Name(), getTensorShape(input), input.Type(), input.QuantizationParams())
		fillInput(input, img)

		// inference
		status := interpreter.Invoke()
		log.Printf("status: %v", status)
		if status != tflite.OK {
			log.Println("invoke failed")
			return
		}

		// print output tensor
		for idx := 0; idx < interpreter.GetOutputTensorCount(); idx++ {
			tensor := interpreter.GetOutputTensor(idx)
			log.Println("output:", tensor.Name(), getTensorShape(tensor), tensor.Type(), tensor.QuantizationParams())
		}
		// convert output
		output := interpreter.GetOutputTensor(0)
		bboxes, confidences, classes := extractOutput(output, float32(*scoreTh), float32(img.Cols()), float32(img.Rows()))

		// NMS
		items := filterOutput(bboxes, confidences, classes, float32(*scoreTh), float32(*nmsTh), labels)

		// build answer
		w.Header().Add("Content-Type", "text/plain")
		bytes, _ := json.Marshal(items)
		w.Write(bytes)
	})
	e := http.ListenAndServe(":8080", nil)
	if e != nil {
		log.Println("Error:", e.Error())
	}
}
