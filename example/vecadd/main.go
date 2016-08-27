package main

/*
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -lcudart
#cgo CFLAGS: -I/usr/local/cuda/include/
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>
*/
import "C"

import (
	"fmt"
	"unsafe"
	// "github.com/pkg/profile"

	"github.com/sunyifan112358/izumo"
)

var length int = 1024
var threadPerBlock int = 512

func main() {
	var v1, v2, gpuSum, cpuSum []float32
	v1 = make([]float32, length)
	v2 = make([]float32, length)
	gpuSum = make([]float32, length)
	cpuSum = make([]float32, length)

	initializeVector(v1, v2)
	gpuVectorAdd(v1, v2, gpuSum)
	cpuVectorAdd(v1, v2, cpuSum)
	verify(gpuSum, cpuSum)

}

func initializeVector(v1, v2 []float32) {
	for i := 0; i < length; i++ {
		v1[i] = float32(1)
		v2[i] = float32(i)
	}
}

func gpuVectorAdd(v1, v2, sum []float32) {
	gV1 := izumo.NewGpuMem(length * 4)
	gV2 := izumo.NewGpuMem(length * 4)
	//gSum := izumo.NewGpuMem(length * 4)

	gV1.CopyHostToDevice(unsafe.Pointer(&v1[0]))
	gV2.CopyHostToDevice(unsafe.Pointer(&v2[0]))

	gV2.CopyDeviceToHost(unsafe.Pointer(&sum[0]))
	fmt.Println(sum)
}

func cpuVectorAdd(v1, v2, sum []float32) {
	for i := 0; i < length; i++ {
		sum[i] = v1[i] + v2[i]
	}
}

func verify(gpuSum, cpuSum []float32) {
	for i := 0; i < length; i++ {
		if gpuSum[i] != cpuSum[i] {
			fmt.Printf("Mismatch at %d, expected %f, but get %f\n", i,
				cpuSum[i], gpuSum[i])
			break
		}
	}
}
