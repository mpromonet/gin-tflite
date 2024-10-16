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
	"bufio"
	"flag"
	"fmt"
	"image"
	"io"
	"log"
	"net/http"
	"os"
	"sort"

	"github.com/gin-contrib/static"
	"github.com/gin-gonic/gin"
	"github.com/mattn/go-tflite/delegates/edgetpu"

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

func getImage(r io.Reader) (gocv.Mat, error) {
	var img gocv.Mat
	// read data
	fileBytes, err := io.ReadAll(r)
	if err != nil {
		fmt.Println(err)
	} else {
		// decode image
		img, err = gocv.IMDecode(fileBytes, gocv.IMReadUnchanged)
		if err != nil {
			log.Println(err)
		} else {
			log.Printf("input size:%vx%v", img.Cols(), img.Rows())
		}
	}

	return img, err
}

func invokeHandler(c *gin.Context, in chan gocv.Mat, out chan []item) {
	log.Println("Header:", c.Request.Header)

	body := c.Request.Body
	if body != nil {
		img, err := getImage(body)
		defer body.Close()

		if err == nil {
			in <- img

			items, ok := <-out
			log.Println("items:", items, "ok:", ok)
			if ok {
				c.JSON(http.StatusOK, items)
			} else {
				c.JSON(http.StatusBadRequest, "cannot get answer from channel")
			}
		} else {
			c.JSON(http.StatusBadRequest, "cannot get image from body")
		}
	} else {
		c.JSON(http.StatusBadRequest, "body is empty")
	}
}

func main() {
	labelPath := flag.String("label", "models/coco.names", "path to label file")
	scoreTh := flag.Float64("score", 0.5, "score threshold")
	nmsTh := flag.Float64("nms", 0.5, "nms threshold")

	flag.Parse()

	labels, err := loadLabels(*labelPath)
	if err != nil {
		log.Fatalf("cannot load labels err:%v", err.Error())
	}

	edgetpuVersion, err := edgetpu.Version()
	if err != nil {
		log.Printf("Could not get EdgeTPU version: %v", err)
	}
	fmt.Printf("EdgeTPU Version: %s\n", edgetpuVersion)

	router := gin.Default()
	router.Use(static.Serve("/", static.LocalFile("./static", false)))

	modelPathArray := flag.Args()
	if len(modelPathArray) == 0 {
		modelPathArray = append(modelPathArray, "models/yolo/lite-model_yolo-v5-tflite_tflite_model_1.tflite")
	}
	modelList := map[string]*Model{}
	for _, modelPath := range modelPathArray {

		model := NewModel(modelPath)
		if model == nil {
			log.Printf("cannot create interpreter model:%v", modelPath)
		} else {
			modelList[modelPath] = model

			in := make(chan gocv.Mat)
			out := make(chan []item)
			go model.modelWorker(float32(*scoreTh), float32(*nmsTh), labels, in, out)
			router.POST("/invoke/"+modelPath, func(c *gin.Context) { invokeHandler(c, in, out) })
		}
	}

	router.GET("/models", func(c *gin.Context) {
		models := []string{}
		for modelName := range modelList {
			models = append(models, modelName)
		}
		sort.Strings(models)
		c.JSON(http.StatusOK, models)
	})

	router.GET("/devices", func(c *gin.Context) {
		devices, err := edgetpu.DeviceList()
		if err != nil {
			log.Printf("Could not get EdgeTPU devices: %v", err)
		} else {
			c.JSON(http.StatusOK, devices)
		}
	})

	// start http server
	e := router.Run(":8080")
	if e != nil {
		log.Println("Error:", e.Error())
	}
}
