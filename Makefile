gin-tflite: /usr/local/include/opencv4/opencv2/cvconfig.h lib/libtensorflowlite_c.so lib/libedgetpu.so $(wildcard *.go)
	CGO_CFLAGS="-I$(PWD)/tensorflow -I$(PWD)/edgetpu/libedgetpu" CGO_LDFLAGS="-L$(PWD)/lib -Wl,-rpath=\$$ORIGIN/lib" go build

/usr/local/include/opencv4/opencv2/cvconfig.h:
	CGO_CFLAGS=-I$(PWD)/tensorflow go get -d
	cd ${HOME}/go/pkg/mod/gocv.io/x/gocv@* && sudo make install

lib:
	mkdir -p lib

edgetpu/libedgetpu/direct/k8/libedgetpu.so.1:
	git submodule update --init edgetpu

lib/libedgetpu.so: lib edgetpu/libedgetpu/direct/k8/libedgetpu.so.1
	cp edgetpu/libedgetpu/direct/k8/libedgetpu.so.1 lib/
	ln -fs libedgetpu.so.1 lib/libedgetpu.so

tflite_build:
	mkdir -p tflite_build

tflite_build/Makefile: tflite_build
	git submodule update --init tensorflow

tflite_build/libtensorflowlite_c.so: tflite_build/Makefile
	cd tflite_build && cmake ../tensorflow/tensorflow/lite/c && make 

lib/libtensorflowlite_c.so: lib tflite_build/libtensorflowlite_c.so
	cp tflite_build/libtensorflowlite_c.so lib/
