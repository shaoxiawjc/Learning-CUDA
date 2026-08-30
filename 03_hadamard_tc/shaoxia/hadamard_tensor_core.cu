#include <cuda_runtime.h>

#include "utils.h"

typedef uint32_t b32;
typedef uint16_t b16;


template <
    typename T,
    const int NUM_CHUNK,
    const int COLS>
__global__ void hadamard_tc_kernel(
    const T *__restrict__ inputs,
    const T *__restrict__ outputs,
    const int rows)
{
    constexpr int ELEMENTS_PER_WARP = 256 * NUM_CHUNK;
    constexpr int WARPS_PER_ROW = COLS / ELEMENTS_PER_WARP;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int row = blockIdx.x;
    if (row > rows)
        return;

    __shared__ b32 sA[WARPS_PER_ROW][NUM_CHUNK][128];

    int g_offset = row * COLS + warp_id * ELEMENTS_PER_WARP;

    #pragma unroll
    for (int i = 0; i < NUM_CHUNK; ++i) {
        cp_async_16B(
            (void*)(sA + i * NUM_CHUNK + lane_id * 4),
            (const void*)(inputs + g_offset + i * NUM_CHUNK + lane_id * 4)
        )
        cp_async_commit();
    }

    b32 reg_A[4];
    for (int i = 0 ; i < NUM_CHUNK ; ++i){
        cp_async_wait<NUM_CHUNK - i - 1>();
        int group_id = laneid >> 2;
        int tid_in_group = laneid % 4;
        reg_A[0] = sA[warp_id][i][group_id * 16 + tid_in_group];
        reg_A[1] = sA[warp_id][i][ * 16 + tid_in_group];
        reg_A[2] = sA[warp_id][i][group_id * 16 + (tid_in_group + 8)];
        reg_A[3] = sA[warp_id][i][(group_id + 8) * 16 + (tid_in_group + 8)];
        
    }

}