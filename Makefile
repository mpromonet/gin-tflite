main: tflite_build/libtensorflowlite_c.so
	CGO_CFLAGS=-I$(PWD)/tensorflow CGO_LDFLAGS=-L$(PWD)/tflite_build go build

tflite_build:
	mkdir -p tflite_build

tflite_build/Makefile: tflite_build
	git submodule update --init

tflite_build/libtensorflowlite_c.so: tflite_build/Makefile
	cd tflite_build && cmake ../tensorflow/tensorflow/lite/c && make

