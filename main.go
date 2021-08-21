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
	"sort"

	"github.com/mattn/go-tflite"

	"gocv.io/x/gocv"
)

var (
	modelPath = flag.String("model", "lite-model_yolo-v5-tflite_tflite_model_1.tflite", "path to model file")
	labelPath = flag.String("label", "coco.names", "path to label file")
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

func min(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

func max(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

func calcIntersectionOverUnion(f1, f2 item) float32 {
	xmin1 := min(f1.X1, f1.X2)
	ymin1 := min(f1.Y1, f1.Y2)
	xmax1 := max(f1.X1, f1.X2)
	ymax1 := max(f1.Y1, f1.Y2)
	xmin2 := min(f2.X1, f2.X2)
	ymin2 := min(f2.Y1, f2.Y2)
	xmax2 := max(f2.X1, f2.X2)
	ymax2 := max(f2.Y1, f2.Y2)

	area1 := (ymax1 - ymin1) * (xmax1 - xmin1)
	area2 := (ymax2 - ymin2) * (xmax2 - xmin2)
	if area1 <= 0 || area2 <= 0 {
		return 0.0
	}

	ixmin := max(xmin1, xmin2)
	iymin := max(ymin1, ymin2)
	ixmax := min(xmax1, xmax2)
	iymax := min(ymax1, ymax2)

	iarea := max(iymax-iymin, 0.0) * max(ixmax-ixmin, 0.0)

	return iarea / (area1 + area2 - iarea)
}

func omitItems(items []item) []item {
	var result []item

	sort.Slice(items, func(i, j int) bool {
		return items[i].Score < items[j].Score
	})

	for _, f1 := range items {
		ignore := false
		for _, f2 := range result {
			iou := calcIntersectionOverUnion(f1, f2)
			if iou >= 0.3 {
				ignore = true
				break
			}
		}

		if !ignore {
			result = append(result, f1)
			if len(result) > 20 {
				break
			}
		}
	}
	return result
}

type item struct {
	X1, Y1, X2, Y2 float32
	Score          float32
	ClassID        int
	ClassName      string
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
	_ = labels

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
		var fileBytes []byte
		file, handler, err := r.FormFile("file")
		if err == nil {
			log.Println("File Size:", handler.Size, " MIME Header:", handler.Header)

			fileBytes, err = ioutil.ReadAll(file)
			if err != nil {
				log.Println(err)
			}
			file.Close()

			img, err := gocv.IMDecode(fileBytes, gocv.IMReadUnchanged)
			if err != nil {
				log.Println(err)
			}

			input := interpreter.GetInputTensor(0)
			output := interpreter.GetOutputTensor(0)
			shape := []int{}
			for idx := 0; idx < output.NumDims(); idx++ {
				shape = append(shape, output.Dim(idx))
			}
			log.Printf("shape: %v", shape)

			resized := gocv.NewMat()
			if input.Type() == tflite.Float32 {
				img.ConvertTo(&resized, gocv.MatTypeCV32F)
				gocv.Resize(resized, &resized, image.Pt(wanted_width, wanted_height), 0, 0, gocv.InterpolationDefault)
				ff, err := resized.DataPtrFloat32()
				if err != nil {
					fmt.Println(err)
				}
				for i := 0; i < len(ff); i++ {
					ff[i] = (ff[i] - 127.5) / 127.5
				}
				copy(input.Float32s(), ff)
			}
			resized.Close()
			status := interpreter.Invoke()
			log.Printf("status: %v", status)
			if status != tflite.OK {
				log.Println("invoke failed")
				return
			}

			// convert
			var loc []float32
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

			var items []item
			if len(loc) != 0 {
				for i := 0; i < shape[1]; i++ {
					idx := (i * shape[2])
					if loc[idx+4] > 0.3 {
						x1 := loc[idx+0]
						y1 := loc[idx+1]
						w := loc[idx+2]
						h := loc[idx+3]
						classId := argmax(loc[idx+5 : idx+85])
						items = append(items, item{
							X1:        x1 - w/2,
							Y1:        y1 - h/2,
							X2:        x1 + w/2,
							Y2:        y1 + h/2,
							Score:     loc[idx+4],
							ClassID:   classId,
							ClassName: getLabel(labels, classId),
						})
					}
				}
			}
			items = omitItems(items)

			w.Header().Add("Content-Type", "text/plain")
			bytes, _ := json.Marshal(items)
			w.Write(bytes)
		}
	})
	e := http.ListenAndServe(":8080", nil)
	if e != nil {
		log.Println("Error:", e.Error())
	}
}
