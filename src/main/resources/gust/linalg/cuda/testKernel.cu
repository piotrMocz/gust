__device__ float f(float x);

extern "C"
__global__ void map_fun(float *A, int n)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= n) return;

    A[idx] = f(A[idx]);
}


__device__ float f(float x) { return x*x - 10.0; }