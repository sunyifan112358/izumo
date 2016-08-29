package izumo

/*
#include <cuda.h>
#include <cuda_runtime.h>
*/
import "C"

// A Stream is a wrapper of the CUStream object. It represents a cuda stream.
type Stream struct {
	cudaStream *C.CUstream
}

// NewStream creates a new stream
func NewStream() (stream *Stream, err *Error) {
	res := C.cuStreamCreate(stream.cudaStream, C.CU_STREAM_DEFAULT)
	if res != C.CUDA_SUCCESS {
		err = NewDriverError(int(res))
		return
	}
	return
}

// Synchronize guarantees all tasks in the stream are finished before return
func (s *Stream) Synchronize() (err *Error) {
	res := C.cuStreamSynchronize(*s.cudaStream)
	if res != C.CUDA_SUCCESS {
		err = NewDriverError(int(res))
		return
	}
	return
}
