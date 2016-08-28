package izumo

import "testing"

func TestGetErrorNameForDriverError(t *testing.T) {
	err := new(Error)
	err.errorCode = 0
	err.isDriverError = true

	name := err.GetErrorName()

	if name != "CUDA_SUCCESS" {
		t.Errorf("Expected %s, but get %s", "CUDA_SUCCESS", name)
	}
}

func TestGetErrorNameForRuntimeError(t *testing.T) {
	err := new(Error)
	err.errorCode = 0
	err.isDriverError = false

	name := err.GetErrorName()

	if name != "cudaSuccess" {
		t.Errorf("Expected %s, but get %s", "cudaSuccess", name)
	}

}
