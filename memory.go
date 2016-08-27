package izumo

/*
#include <cuda.h>
#include <cuda_runtime.h>
*/
import "C"

import (
	"fmt"
	"os"
	"unsafe"
)

// A GpuMem is a piece of memory that is allocated on GPU
type GpuMem struct {
	cudaPointer unsafe.Pointer
	size        int
}

// Creates a GpuMem object and allocate memory on a GPU
func NewGpuMem(size int) (gpuMem *GpuMem) {
	gpuMem = new(GpuMem)
	gpuMem.size = size

	err := C.cudaMalloc(&gpuMem.cudaPointer, C.size_t(size))
	if err != C.cudaSuccess {
		fmt.Println(err)
		os.Exit(-1)
	}
	return
}

// Copy memory from device to host
func (m *GpuMem) CopyDeviceToHost(hostPtr unsafe.Pointer) {
	err := C.cudaMemcpy(hostPtr, m.cudaPointer, C.size_t(m.size), C.cudaMemcpyDeviceToHost)
	if err != C.cudaSuccess {
		fmt.Println(err)
		os.Exit(-1)
	}

}

// Copy memory from host to device
func (m *GpuMem) CopyHostToDevice(hostPtr unsafe.Pointer) {
	err := C.cudaMemcpy(m.cudaPointer, hostPtr, C.size_t(m.size), C.cudaMemcpyHostToDevice)
	if err != C.cudaSuccess {
		fmt.Println(err)
		os.Exit(-1)
	}
}
