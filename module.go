package izumo

//#include <cuda.h>
import "C"

import (
	"fmt"
	"os"
)

// A Module is a wrapper for the cuda module. It contains all the information
// that is contained in a PTX file or cubin file.
type Module struct {
	cudaModule C.CUmodule
}

// LoadModuleFromFile create a module and load the module from ptx file or
// cubin file
func LoadModuleFromFile(moduleFilePath string) (module *Module) {
	module = new(Module)
	err := C.cuModuleLoad(&module.cudaModule, C.CString(moduleFilePath))
	if err != C.CUDA_SUCCESS {
		fmt.Println(err)
		os.Exit(-1)
	}
	return
}
