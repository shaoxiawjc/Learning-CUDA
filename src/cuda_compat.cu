#include <cuda_runtime.h>

extern "C" cudaError_t cudaGetDeviceProperties_v2(struct cudaDeviceProp *prop, int device) {
    return cudaGetDeviceProperties(prop, device);
}
