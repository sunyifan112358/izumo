package izumo

//#include <cuda.h>
import "C"

// A Module is a wrapper for the cuda module. It contains all the information
// that is contained in a PTX file or cubin file.
type Module struct {
	cudaModule C.CUmodule
}

// LoadModuleFromFile create a module and load the module from ptx file or
// cubin file
func LoadModuleFromFile(moduleFilePath string) (module *Module, err *Error) {
	module = new(Module)
	res := C.cuModuleLoad(&module.cudaModule, C.CString(moduleFilePath))
	if res != C.CUDA_SUCCESS {
		err = NewDriverError(int(res))
		return
	}
	return
}
