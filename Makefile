gin-tflite: tflite_build/libtensorflowlite_c.so /usr/local/include/opencv4/opencv2/cvconfig.h $(wildcard *.go)
	CGO_CFLAGS=-I$(PWD)/tensorflow CGO_LDFLAGS="-L$(PWD)/tflite_build -L/usr/local/lib" go build

/usr/local/include/opencv4/opencv2/cvconfig.h:
	CGO_CFLAGS=-I$(PWD)/tensorflow go get -d
	cd ${HOME}/go/pkg/mod/gocv.io/x/gocv@* && sudo make install

tflite_build:
	mkdir -p tflite_build

tflite_build/Makefile: tflite_build
	git submodule update --init

tflite_build/libtensorflowlite_c.so: tflite_build/Makefile
	cd tflite_build && cmake ../tensorflow/tensorflow/lite/c -DCMAKE_INSTALL_PREFIX=/usr/local && make

