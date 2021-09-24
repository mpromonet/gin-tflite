gin-tflite: tflite_build/libtensorflowlite_c.so /usr/local/include/opencv4/opencv2/cvconfig.h edgetpu/libedgetpu/direct/k8/libedgetpu.so $(wildcard *.go)
	CGO_CFLAGS="-I$(PWD)/tensorflow -I$(PWD)/edgetpu/libedgetpu" CGO_LDFLAGS="-L$(PWD)/tflite_build -Wl,-rpath=\$$ORIGIN/tflite_build -L$(PWD)/edgetpu/libedgetpu/direct/k8 -Wl,-rpath=\$$ORIGIN/edgetpu/libedgetpu/direct/k8" go build

edgetpu/libedgetpu/direct/k8/libedgetpu.so:
	git submodule update --init edgetpu
	ln -s libedgetpu.so.1 edgetpu/libedgetpu/direct/k8/libedgetpu.so

/usr/local/include/opencv4/opencv2/cvconfig.h:
	CGO_CFLAGS=-I$(PWD)/tensorflow go get -d
	cd ${HOME}/go/pkg/mod/gocv.io/x/gocv@* && sudo make install

tflite_build:
	mkdir -p tflite_build

tflite_build/Makefile: tflite_build
	git submodule update --init

tflite_build/libtensorflowlite_c.so: tflite_build/Makefile
	cd tflite_build && cmake ../tensorflow/tensorflow/lite/c -DCMAKE_INSTALL_PREFIX=/usr/local && make

