// Compatibility for tester_moore.o built against a newer MUSA runtime ABI.
#include <musa_runtime_api.h>

extern "C" musaError_t musaGetDeviceProperties_v2(
    struct musaDeviceProp* prop, int device) {
  return musaGetDeviceProperties(prop, device);
}
