package izumo

/*
#include <cuda.h>
#include <cuda_runtime.h>
*/
import "C"

import (
	"unsafe"
)

// A GpuMem is a piece of memory that is allocated on GPU
type GpuMem struct {
	cudaPointer unsafe.Pointer
	size        uint
}

// NewGpuMem creates a GpuMem object and allocate memory on a GPU
func NewGpuMem(size uint) (gpuMem *GpuMem, err *Error) {
	gpuMem = new(GpuMem)
	gpuMem.size = size

	res := C.cudaMalloc(&gpuMem.cudaPointer, C.size_t(size))
	if res != C.cudaSuccess {
		err = NewRuntimeError(int(res))
		return
	}
	return
}

// CopyDeviceToHost copies memory from device to host
func (m *GpuMem) CopyDeviceToHost(hostPtr unsafe.Pointer) (err *Error) {
	res := C.cudaMemcpy(hostPtr, m.cudaPointer, C.size_t(m.size), C.cudaMemcpyDeviceToHost)
	if res != C.cudaSuccess {
		err = NewRuntimeError(int(res))
	}
	return
}

// CopyHostToDevice copies memory from host to device
func (m *GpuMem) CopyHostToDevice(hostPtr unsafe.Pointer) (err *Error) {
	res := C.cudaMemcpy(m.cudaPointer, hostPtr, C.size_t(m.size), C.cudaMemcpyHostToDevice)
	if res != C.cudaSuccess {
		err = NewRuntimeError(int(res))
	}
	return
}

// GetGpuPointer returns the pointer that points to the memory allocaed on
// the GPU.
func (m *GpuMem) GetGpuPointer() (ptr unsafe.Pointer) {
	ptr = m.cudaPointer
	return
}

// Free the GPU memory
func (m *GpuMem) Free() (err *Error) {
	res := C.cudaFree(m.cudaPointer)
	if res != C.cudaSuccess {
		err = NewRuntimeError(int(res))
	}
	return
}
