package izumo

/*
#include <cuda.h>
#include <cuda_runtime.h>
*/
import "C"

// Error is a wrapper of the cuda error code
type Error struct {
	errorCode     int
	isDriverError bool
}

// GetErrorName returns the name of the error
func (err *Error) GetErrorName() (name string) {
	if err.isDriverError {
		var cStr *C.char
		C.cuGetErrorName(C.CUresult(err.errorCode), &cStr)
		return C.GoString(cStr)
	}
	cStr := C.cudaGetErrorName(C.cudaError_t(err.errorCode))
	return C.GoString(cStr)
}

// GetErrorString retruns the description sting of the error
func (err *Error) GetErrorString() (desc string) {
	return ""
}
