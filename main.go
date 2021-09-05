package main

import (
	"bufio"
	"flag"
	"fmt"
	"image"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/gin-contrib/static"
	"github.com/gin-gonic/gin"
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

func getImage(r io.Reader) gocv.Mat {
	// read data
	fileBytes, err := ioutil.ReadAll(r)
	if err != nil {
		fmt.Println(err)
	}

	// decode image
	img, err := gocv.IMDecode(fileBytes, gocv.IMReadUnchanged)
	if err != nil {
		log.Println(err)
	}
	log.Printf("input size:%vx%v", img.Cols(), img.Rows())
	return img
}

func modelWorker(interpreter *tflite.Interpreter, scoreTh float32, nmsTh float32, labels []string, in chan gocv.Mat, out chan []item) {
	idleDuration := 10 * time.Millisecond
	timeout := time.NewTimer(idleDuration)
	defer timeout.Stop()
	for {
		timeout.Reset(idleDuration)
		select {
		case img := <-in:
			if !img.Empty() {
				// fill input tensor
				input := interpreter.GetInputTensor(0)
				log.Println("input shape:", input.Name(), getTensorShape(input), input.Type(), input.QuantizationParams())
				fillInput(input, img)

				// inference
				status := interpreter.Invoke()
				log.Printf("status: %v", status)
				if status != tflite.OK {
					log.Println("invoke failed")
				}

				// print output tensor
				for idx := 0; idx < interpreter.GetOutputTensorCount(); idx++ {
					tensor := interpreter.GetOutputTensor(idx)
					log.Println("output:", tensor.Name(), getTensorShape(tensor), tensor.Type(), tensor.QuantizationParams())
				}

				// convert output
				output := interpreter.GetOutputTensor(0)
				bboxes, confidences, classes := extractOutput(output, scoreTh, float32(img.Cols()), float32(img.Rows()))

				// NMS
				items := filterOutput(bboxes, confidences, classes, scoreTh, nmsTh, labels)

				out <- items
				img.Close()
			} else {
				break
			}
		case <-timeout.C:
		}
	}
}

func invokeHandler(c *gin.Context, in chan gocv.Mat, out chan []item) {
	log.Println("Header:", c.Request.Header)

	body := c.Request.Body
	if body != nil {
		img := getImage(body)
		defer body.Close()

		if len(in) == cap(in) {
			oldimg := <-in
			oldimg.Close()
		}

		in <- img

		items := <-out
		log.Println("items:", items)

		c.JSON(http.StatusOK, items)
	} else {
		c.JSON(http.StatusBadRequest, "body is empty")
	}
}

type Model struct {
	model  *tflite.Model
	interp *tflite.Interpreter
}

func main() {
	labelPath := flag.String("label", "models/coco.names", "path to label file")
	scoreTh := flag.Float64("score", 0.3, "score threshold")
	nmsTh := flag.Float64("nms", 0.5, "nms threshold")

	flag.Parse()

	labels, err := loadLabels(*labelPath)
	if err != nil {
		log.Fatalf("cannot load labels err:%v", err.Error())
	}

	router := gin.Default()
	router.Use(static.Serve("/", static.LocalFile("./static", false)))

	modelPathArray := flag.Args()
	if len(modelPathArray) == 0 {
		modelPathArray = append(modelPathArray, "models/lite-model_yolo-v5-tflite_tflite_model_1.tflite")
	}
	modelList := map[string]Model{}
	for _, modelPath := range modelPathArray {

		model, interpreter := createInterpreter(modelPath)
		if interpreter == nil {
			log.Println("cannot create interpreter")
		} else {
			modelList[modelPath] = Model{model, interpreter}

			in := make(chan gocv.Mat, 25)
			out := make(chan []item, 10)
			go modelWorker(interpreter, float32(*scoreTh), float32(*nmsTh), labels, in, out)
			router.POST("/invoke/"+modelPath, func(c *gin.Context) { invokeHandler(c, in, out) })
		}
	}

	router.GET("/models", func(c *gin.Context) {
		models := []string{}
		for modelName := range modelList {
			models = append(models, modelName)
		}
		c.JSON(http.StatusOK, models)
	})

	// start http server
	e := router.Run(":8080")
	if e != nil {
		log.Println("Error:", e.Error())
	}
}
