package izumo

/*
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

static inline void setArgumentPtr(void** argPtrs, int pos, void* arg) {
	void **ptr = malloc(sizeof(void*));
	*ptr = arg;
	argPtrs[pos] = ptr;
}
*/
import "C"

import (
	"log"
	"reflect"
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
			C.setArgumentPtr(argumentsPtrs, C.int(i), gMem.cudaPointer)
		} else {
			log.Fatal(reflect.TypeOf(arg).Name() + "is not currently supported " +
				"as kernel arguments.")
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
