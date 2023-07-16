/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
** -------------------------------------------------------------------------*/

package main

import (
	"image"
	"log"
	"math"

	"github.com/mattn/go-tflite"
)

type YoloPostProcessing struct {
	PostProcessing
}

func (p YoloPostProcessing) extractResult(interp *tflite.Interpreter, scoreTh float32, width float32, height float32) ([]image.Rectangle, []float32, []int) {
	bboxes := []image.Rectangle{}
	confidences := []float32{}
	classes := []int{}
	for idx := 0; idx < interp.GetOutputTensorCount(); idx++ {
		output := interp.GetOutputTensor(idx)
		log.Println("output:", output.Name(), getTensorShape(output), output.Type(), output.QuantizationParams())
		bboxes_, confidences_, classes_ := p.extractBoxesTensor(output, scoreTh, width, height)
		bboxes = append(bboxes, bboxes_...)
		confidences = append(confidences, confidences_...)
		classes = append(classes, classes_...)
	}
	return bboxes, confidences, classes
}

func (y YoloPostProcessing) extractBoxesTensor(output *tflite.Tensor, scoreTh float32, width float32, height float32) ([]image.Rectangle, []float32, []int) {

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
		copy(loc, f)
	}

	bboxes := []image.Rectangle{}
	confidences := []float32{}
	classes := []int{}
	if len(loc) != 0 {
		log.Println("output dims:", output.NumDims())
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
