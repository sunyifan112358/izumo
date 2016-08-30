extern "C" __global__ void add(float *a, float *b, float *c) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  c[index] = a[index] + b[index];
}

