/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
** -------------------------------------------------------------------------*/

package main

import (
	"image"

	"github.com/mattn/go-tflite"
)

type PostProcessing interface {
	extractResult(interp *tflite.Interpreter, scoreTh float32, width float32, height float32) ([]image.Rectangle, []float32, []int)
}
