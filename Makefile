gin-tflite: /usr/local/include/opencv4/opencv2/cvconfig.h lib/libtensorflowlite_c.so $(wildcard *.go)
	go build

/usr/local/include/opencv4/opencv2/cvconfig.h:
	go get -d
	env
	cd ${HOME}/go/pkg/mod/gocv.io/x/gocv@* && make install

lib:
	mkdir -p lib

tflite_build:
	mkdir -p tflite_build

tflite_build/Makefile: tflite_build
	git submodule update --init tensorflow

tflite_build/libtensorflowlite_c.so: tflite_build/Makefile
	cd tflite_build && cmake ../tensorflow/tensorflow/lite/c -DTFLITE_ENABLE_GPU=ON && make 

lib/libtensorflowlite_c.so: lib tflite_build/libtensorflowlite_c.so
	cp tflite_build/libtensorflowlite_c.so lib/
