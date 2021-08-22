package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	_ "image/png"
	"io/ioutil"
	"log"
	"net/http"
	"os"

	"github.com/mattn/go-tflite"

	"gocv.io/x/gocv"
)

var (
	modelPath = flag.String("model", "models/lite-model_yolo-v5-tflite_tflite_model_1.tflite", "path to model file")
	labelPath = flag.String("label", "models/coco.names", "path to label file")
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

func argmax(f []float32) int {
	r, m := 0, f[0]
	for i, v := range f {
		if v > m {
			m = v
			r = i
		}
	}
	return r
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

func main() {
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

	input := interpreter.GetInputTensor(0)
	wanted_height := input.Dim(1)
	wanted_width := input.Dim(2)

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

		img, err := gocv.IMDecode(fileBytes, gocv.IMReadUnchanged)
		if err != nil {
			log.Println(err)
		}

		input := interpreter.GetInputTensor(0)

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

		// inference
		status := interpreter.Invoke()
		log.Printf("status: %v", status)
		if status != tflite.OK {
			log.Println("invoke failed")
			return
		}

		// convert output
		var loc []float32
		output := interpreter.GetOutputTensor(0)
		shape := []int{}
		for idx := 0; idx < output.NumDims(); idx++ {
			shape = append(shape, output.Dim(idx))
		}
		log.Printf("shape: %v", shape)
		outputtype := output.Type()
		log.Printf("type: %v", outputtype)
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

		bboxes := []image.Rectangle{}
		confidences := []float32{}
		classes := []int{}

		var items []item
		if len(loc) != 0 {
			for i := 0; i < shape[1]; i++ {
				idx := (i * shape[2])
				if loc[idx+4] > 0.3 {
					x1 := int(loc[idx+0] * float32(img.Cols()))
					y1 := int(loc[idx+1] * float32(img.Rows()))
					w := int(loc[idx+2] * float32(img.Cols()))
					h := int(loc[idx+3] * float32(img.Rows()))
					bboxes = append(bboxes, image.Rect(x1-w/2, y1-h/2, x1+w/2, y1+h/2))
					confidences = append(confidences, loc[idx+4])
					classId := argmax(loc[idx+5 : idx+85])
					classes = append(classes, classId)
				}
			}
		}
		indices := make([]int, len(bboxes))
		for i := range indices {
			indices[i] = -1
		}
		gocv.NMSBoxes(bboxes, confidences, 0.5, 0.3, indices)

		for _, idx := range indices {
			if idx > 0 {
				classID := classes[idx]
				confidence := confidences[idx]
				bbox := bboxes[idx]
				detect := item{ClassID: classID,
					ClassName: getLabel(labels, classID),
					Score:     confidence,
					Box:       bbox}
				items = append(items, detect)
			}
		}

		w.Header().Add("Content-Type", "text/plain")
		bytes, _ := json.Marshal(items)
		w.Write(bytes)
	})
	e := http.ListenAndServe(":8080", nil)
	if e != nil {
		log.Println("Error:", e.Error())
	}
}
