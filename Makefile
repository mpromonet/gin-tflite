gin-tflite: tflite_build/libtensorflowlite_c.a /usr/local/include/opencv4
	CGO_CFLAGS=-I$(PWD)/tensorflow CGO_LDFLAGS=-L$(PWD)/tflite_build go build

/usr/local/include/opencv4:
	CGO_CFLAGS=-I$(PWD)/tensorflow CGO_LDFLAGS=-L$(PWD)/tflite_build go get -d
	cd ${HOME}/go/pkg/mod/gocv.io/x/gocv@* && sudo make install

tflite_build:
	mkdir -p tflite_build

tflite_build/Makefile: tflite_build
	git submodule update --init

tflite_build/libtensorflowlite_c.a: tflite_build/Makefile
	cd tflite_build && cmake ../tensorflow/tensorflow/lite/c -DTFLITE_C_BUILD_SHARED_LIBS=OFF && make

