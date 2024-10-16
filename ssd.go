/*
 * SPDX-License-Identifier: Unlicense
 *
 * This is free and unencumbered software released into the public domain.
 *
 * Anyone is free to copy, modify, publish, use, compile, sell, or distribute this
 * software, either in source code form or as a compiled binary, for any purpose,
 * commercial or non-commercial, and by any means.
 *
 * For more information, please refer to <http://unlicense.org/>
 */

package main

import (
	"fmt"
	"image"

	"github.com/mattn/go-tflite"
)

type SsdPostProcessing struct {
	PostProcessing
}

func copySlice(f []float32) []float32 {
	ff := make([]float32, len(f), len(f))
	copy(ff, f)
	return ff
}

func (p SsdPostProcessing) extractResult(interp *tflite.Interpreter, scoreTh float32, width float32, height float32) ([]image.Rectangle, []float32, []int) {
	bboxes := []image.Rectangle{}
	confidences := []float32{}
	classes := []int{}

	if interp.GetOutputTensorCount() > 2 {
		l := copySlice(interp.GetOutputTensor(0).Float32s())
		c := copySlice(interp.GetOutputTensor(1).Float32s())
		s := copySlice(interp.GetOutputTensor(2).Float32s())
		fmt.Printf("output: %vx%vx%v\n", len(l), len(c), len(s))
		for idx := 0; 4*idx < len(l) && idx < len(c) && idx < len(s); idx++ {
			bboxes = append(bboxes, image.Rectangle{image.Point{int(l[4*idx] * width), int(l[4*idx+1] * height)}, image.Point{int(l[4*idx+2] * width), int(l[4*idx+3] * height)}})
			confidences = append(confidences, s[idx])
			classes = append(classes, int(c[idx]))
		}
		fmt.Printf("output: %v %v %v\n", bboxes, confidences, classes)
	}

	return bboxes, confidences, classes
}
