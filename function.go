package izumo

/*
#include <cuda.h>
#include <cuda_runtime.h>
*/
import "C"

import (
	"unsafe"
)

// A Function is a wrapper of the CUfunction object. It represents a cuda
// kernel.
type Function struct {
	cudaFunction *C.CUfunction
}

// Dim3 is a collection of the x, y, and z size
type Dim3 struct {
	X, Y, Z int
}

// Start a kernel on a GPU
func (f *Function) LaunchKernel(gridDim Dim3, blockDim Dim3,
	sharedMemBytes int,
	arguments ...unsafe.Pointer) (err *Error) {

	stream, err := NewStream()
	if err != nil {
		return err
	}

	argumentsPtrs := make([]unsafe.Pointer, 10)
	for _, arg := range arguments {
		argumentsPtrs = append(argumentsPtrs, arg)
	}

	res := C.cuLaunchKernel(*f.cudaFunction,
		C.uint(gridDim.X), C.uint(gridDim.Y), C.uint(gridDim.Z),
		C.uint(blockDim.X), C.uint(blockDim.Y), C.uint(blockDim.Z),
		C.uint(sharedMemBytes),
		*stream.cudaStream,
		&argumentsPtrs[0],
		nil)
	if res != C.CUDA_SUCCESS {
		err = NewDriverError(int(res))
		return
	}

	err = stream.Synchronize()
	if err != nil {
		return err
	}

	return

}
