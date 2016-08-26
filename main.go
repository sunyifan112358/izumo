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
)

func main() {
	// defer profile.Start().Stop()

	deviceCount := new(C.int)
	cudaError := C.cudaGetDeviceCount(deviceCount)
	fmt.Println(cudaError, *deviceCount)

	deviceProp := new(C.struct_cudaDeviceProp)
	C.cudaGetDeviceProperties(deviceProp, 0)
	fmt.Println(*deviceProp)
	fmt.Println(string(C.GoBytes(unsafe.Pointer(&(*deviceProp).name), 256)))
}
