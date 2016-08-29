package izumo

import "testing"

func TestGetErrorNameForDriverError(t *testing.T) {
	err := NewDriverError(0)

	name := err.GetErrorName()

	if name != "CUDA_SUCCESS" {
		t.Errorf("Expected %s, but get %s", "CUDA_SUCCESS", name)
	}
}

func TestGetErrorNameForRuntimeError(t *testing.T) {
	err := NewRuntimeError(0)

	name := err.GetErrorName()

	if name != "cudaSuccess" {
		t.Errorf("Expected %s, but get %s", "cudaSuccess", name)
	}
}

func TestGetErrorStringForDriverError(t *testing.T) {
	err := NewDriverError(0)

	name := err.GetErrorString()

	if name != "no error" {
		t.Errorf("Expected %s, but get %s", "no error", name)
	}
}

func TestGetErrorStringForRuntimeError(t *testing.T) {
	err := NewRuntimeError(0)

	name := err.GetErrorString()

	if name != "no error" {
		t.Errorf("Expected %s, but get %s", "no error", name)
	}
}
