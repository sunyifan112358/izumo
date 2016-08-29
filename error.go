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

// Create an instance of a runtime error
func NewRuntimeError(code int) (err *Error) {
	err = new(Error)
	err.errorCode = code
	err.isDriverError = false
	return
}

// Create an instance of a driver error
func NewDriverError(code int) (err *Error) {
	err = new(Error)
	err.errorCode = code
	err.isDriverError = true
	return
}

// GetErrorName returns the name of the error
func (err *Error) GetErrorName() (name string) {
	var cStr *C.char
	if err.isDriverError {
		C.cuGetErrorName(C.CUresult(err.errorCode), &cStr)
	} else {
		cStr = C.cudaGetErrorName(C.cudaError_t(err.errorCode))
	}
	return C.GoString(cStr)
}

// GetErrorString retruns the description sting of the error
func (err *Error) GetErrorString() (desc string) {
	var cStr *C.char
	if err.isDriverError {
		C.cuGetErrorString(C.CUresult(err.errorCode), &cStr)
	} else {
		cStr = C.cudaGetErrorString(C.cudaError_t(err.errorCode))
	}
	return C.GoString(cStr)
}
