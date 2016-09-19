package izumo

/*
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>

static void **allocateArgumentPtrs(int numArgs) {
	int i;
	void **arguments = malloc(sizeof(void*) * numArgs);
	for (i = 0; i < numArgs; i++) {
		arguments[i] = malloc(sizeof(void*));
	}
	return arguments;
}

static inline void setArgument(void** argPtrs, int pos, void* arg, int size) {
	void *ptr = malloc(size);
	memcpy(ptr, arg, size);
	argPtrs[pos] = ptr;
}
*/
import "C"

import (
	"unsafe"
)

// A Function is a wrapper of the CUfunction object. It represents a cuda
// kernel.
type Function struct {
	cudaFunction C.CUfunction
}

// Dim3 is a collection of the x, y, and z size
type Dim3 struct {
	X, Y, Z int
}

// Start a kernel on a GPU
func (f *Function) LaunchKernel(
	gridDim Dim3, blockDim Dim3,
	sharedMemBytes int,
	arguments ...interface{}) (err *Error) {

	stream, err := NewStream()
	if err != nil {
		return err
	}

	argumentsPtrs := C.allocateArgumentPtrs(C.int(len(arguments)))
	for i, arg := range arguments {
		if gMem, ok := arg.(GpuMem); ok {
			C.setArgument(argumentsPtrs, C.int(i), unsafe.Pointer(&gMem.cudaPointer), 8)
		} else if data, ok := arg.([]byte); ok {
			C.setArgument(argumentsPtrs, C.int(i), unsafe.Pointer(&data[0]), C.int(len(data)))
		} else {
			C.setArgument(argumentsPtrs, C.int(i), unsafe.Pointer(&arg), C.int(unsafe.Sizeof(arg)))
		}
	}

	res := C.cuLaunchKernel(f.cudaFunction,
		C.uint(gridDim.X), C.uint(gridDim.Y), C.uint(gridDim.Z),
		C.uint(blockDim.X), C.uint(blockDim.Y), C.uint(blockDim.Z),
		C.uint(sharedMemBytes),
		stream.cudaStream,
		argumentsPtrs,
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
