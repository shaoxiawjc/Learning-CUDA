#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cassert>
#include <type_traits>

#include "utils.h"

typedef uint32_t b32;
typedef uint16_t b16;


// 计算 H16 矩阵在当前 lane 上对应的数据
// 作为 MMA 的 A 矩阵，有 8 个 b16 的数据，而他们是事先定义好的
// p1 和 p2 是 sign-bit tables，pp、pn、np、nn 是 packed +/-1.0
// 每一个 lane 可以通过查找 p1 和 p2 来确定自己对应的 sign，然后根据 sign 来选择 pp、pn、np、nn 中的一个作为自己的数据
template <typename T>
__device__ __forceinline__ void make_h16_a_frag(b32 (&frag)[4], int lane_id)
{
    static_assert(std::is_same<T, __half>::value ||
                      std::is_same<T, __nv_bfloat16>::value,
                  "T must be __half or __nv_bfloat16");

    constexpr b32 p1[4] = {0xFFAACC99u, 0xFFAACC99u, 0xFFAACC99u, 0x00553366u};
    constexpr b32 p2[4] = {0xF0A5C396u, 0xF0A5C396u, 0xF0A5C396u, 0x0F5A3C69u};

    b32 pp, pn, np, nn;
    if constexpr (std::is_same<T, __half>::value) {
        pp = 0x3C003C00u; pn = 0xBC003C00u; np = 0x3C00BC00u; nn = 0xBC00BC00u;
    } else {
        pp = 0x3F803F80u; pn = 0xBF803F80u; np = 0x3F80BF80u; nn = 0xBF80BF80u;
    }

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        bool lo = (p1[j] >> (31 - lane_id)) & 1u;  // even-column element sign
        bool hi = (p2[j] >> (31 - lane_id)) & 1u;  // odd-column element sign
        frag[j] = lo ? (hi ? pp : pn) : (hi ? np : nn);
    }
}


// mma.m16n8k16: D = A @ B + 0. A is 16x16 row-major (4 b32), B is 16x8 col-major (2 b32),
// D is 16x8 (2 b32). Both fp16 and bf16 accumulate in f32, then cvt back to 2 b32.
template <typename T>
__device__ __forceinline__ void mma_m16_n8_k16(
    b32 a0, b32 a1, b32 a2, b32 a3,
    b32 b0, b32 b1,
    b32 &c0, b32 &c1)
{
    static_assert(std::is_same<T, __half>::value ||
                      std::is_same<T, __nv_bfloat16>::value,
                  "T must be __half or __nv_bfloat16");

    b32 temp0, temp1, temp2, temp3;
    const b32 zero = 0;
    if constexpr (std::is_same<T, __half>::value)
    {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=r"(temp0), "=r"(temp1), "=r"(temp2), "=r"(temp3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1),
              "r"(zero), "r"(zero), "r"(zero), "r"(zero));
        asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(c0) : "r"(temp1), "r"(temp0));
        asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(c1) : "r"(temp3), "r"(temp2));
    }
    else
    {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=r"(temp0), "=r"(temp1), "=r"(temp2), "=r"(temp3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1),
              "r"(zero), "r"(zero), "r"(zero), "r"(zero));
        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(c0) : "r"(temp1), "r"(temp0));
        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(c1) : "r"(temp3), "r"(temp2));
    }
}


// Reassemble the chunk index from a digit `d` (occupying bits [b, b+s)) and the
// rest `r` (the index with those s bits removed).
__device__ __forceinline__ int chunk_index(int b, int s, int d, int r)
{
    return (r & ((1 << b) - 1)) | (d << b) | ((r >> b) << (b + s));
}

// Read the b16 element at position `pos` (0..255) of the chunk identified by
// (digit d, rest r). Returns 0 for d >= dsize so an s<4 digit zero-pads the
// mma's K dimension (H16's top-left 2^s block is H_{2^s}).
__device__ __forceinline__ b16 stage_read(const b32 *sA_flat, int b, int s, int dsize, int d, int r, int pos)
{
    if (d >= dsize)
        return 0;
    b32 v = sA_flat[chunk_index(b, s, d, r) * 128 + (pos >> 1)];
    return (pos & 1) ? (b16)(v >> 16) : (b16)(v & 0xFFFF);
}


template <
    typename T,
    const int NUM_CHUNK,
    const int LOG_COLS>
__global__ void hadamard_tc_kernel(
    const T *__restrict__ inputs,
    T *__restrict__ outputs,
    const int rows)
{
    constexpr int COLS = 1 << LOG_COLS;

    if constexpr (LOG_COLS < 8)
    {
        const int row = blockIdx.x;
        if (row >= rows)
            return;

        __shared__ float s[COLS];
        const T *in = inputs + row * COLS;
        T *out = outputs + row * COLS;
        const int tid = threadIdx.x;

        for (int i = tid; i < COLS; i += blockDim.x)
        {
            float v;
            if constexpr (std::is_same<T, __half>::value)
                v = __half2float(in[i]);
            else
                v = __bfloat162float(in[i]);
            s[i] = v;
        }
        __syncthreads();

        for (int h = 1; h < COLS; h <<= 1)
        {
            for (int i = tid; i < COLS; i += blockDim.x)
            {
                if ((i & h) == 0)
                {
                    float a = s[i], b = s[i + h];
                    s[i] = a + b;
                    s[i + h] = a - b;
                }
            }
            __syncthreads();
        }

        for (int i = tid; i < COLS; i += blockDim.x)
        {
            if constexpr (std::is_same<T, __half>::value)
                out[i] = __float2half(s[i]);
            else
                out[i] = __float2bfloat16(s[i]);
        }
        return;
    }

    constexpr int ELEMENTS_PER_WARP = 256 * NUM_CHUNK;
    constexpr int WARPS_PER_ROW = COLS / ELEMENTS_PER_WARP;
    constexpr int LOG_NCHUNK = LOG_COLS - 8;

    // Each +1 in LOG_COLS doubles the row length and hence the shared memory
    // (COLS * 2 bytes): LOG_COLS=15 -> 64KB (needs the opt-in shared-memory
    // limit on sm_80).
    static_assert(LOG_COLS >= 1 && LOG_COLS <= 15,
                  "LOG_COLS must be in [1, 15] (COLS in [2, 32768])");
    static_assert(NUM_CHUNK <= 8,
                  "NUM_CHUNK must be <= 8: cp.async tracks 8 groups");

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int row = blockIdx.x;
    if (row >= rows)
        return;

    // Dynamic shared memory (host must size it WARPS_PER_ROW*NUM_CHUNK*128*4 bytes
    // and, for LOG_COLS >= 15, opt in via cudaFuncAttributeMaxDynamicSharedMemorySize).
    extern __shared__ b32 sA[];

    int g_offset = row * COLS + warp_id * ELEMENTS_PER_WARP;

    #pragma unroll
    for (int i = 0; i < NUM_CHUNK; ++i) {
        cp_async_16B(
            (void*)&sA[(warp_id * NUM_CHUNK + i) * 128 + lane_id * 4],
            (const void*)(inputs + g_offset + i * 256 + lane_id * 8)
        );
        cp_async_commit();
    }
    b32 h_frag[4];
    make_h16_a_frag<T>(h_frag, lane_id);
    b32 x_frag[4];

    // Software pipeline: each chunk is its own cp.async group, so wait only until
    // chunk i has landed, leaving chunks i+1..NUM_CHUNK-1 still in flight.
    #pragma unroll
    for (int i = 0; i < NUM_CHUNK; ++i) {
        switch (i) {
            case 0: asm volatile("cp.async.wait_group %0;\n" ::"n"(NUM_CHUNK - 1)); break;
            case 1: asm volatile("cp.async.wait_group %0;\n" ::"n"(NUM_CHUNK - 2)); break;
            case 2: asm volatile("cp.async.wait_group %0;\n" ::"n"(NUM_CHUNK - 3)); break;
            case 3: asm volatile("cp.async.wait_group %0;\n" ::"n"(NUM_CHUNK - 4)); break;
            case 4: asm volatile("cp.async.wait_group %0;\n" ::"n"(NUM_CHUNK - 5)); break;
            case 5: asm volatile("cp.async.wait_group %0;\n" ::"n"(NUM_CHUNK - 6)); break;
            case 6: asm volatile("cp.async.wait_group %0;\n" ::"n"(NUM_CHUNK - 7)); break;
            case 7: asm volatile("cp.async.wait_group %0;\n" ::"n"(NUM_CHUNK - 8)); break;
        }
        int group_id = lane_id >> 2;
        int tid_in_group = lane_id % 4;
        int sA_base = (warp_id * NUM_CHUNK + i) * 128;
        x_frag[0] = sA[sA_base + group_id * 8 + tid_in_group];
        x_frag[1] = sA[sA_base + group_id * 8 + tid_in_group + 4];
        x_frag[2] = sA[sA_base + (group_id + 8) * 8 + tid_in_group];
        x_frag[3] = sA[sA_base + (group_id + 8) * 8 + tid_in_group + 4];

        b32 c_frag[4];
        mma_m16_n8_k16<T>(h_frag[0], h_frag[1], h_frag[2], h_frag[3],
                          x_frag[0], x_frag[1], c_frag[0], c_frag[1]);  // cols 0..7
        mma_m16_n8_k16<T>(h_frag[0], h_frag[1], h_frag[2], h_frag[3],
                          x_frag[2], x_frag[3], c_frag[2], c_frag[3]);  // cols 8..15


        b32 tmp = c_frag[1];
        c_frag[1] = c_frag[2];
        c_frag[2] = tmp;

        b32 d_frag[4];
        mma_m16_n8_k16<T>(h_frag[0], h_frag[1], h_frag[2], h_frag[3],
                          c_frag[0], c_frag[1], d_frag[0], d_frag[1]);  // cols 0..7
        mma_m16_n8_k16<T>(h_frag[0], h_frag[1], h_frag[2], h_frag[3],
                          c_frag[2], c_frag[3], d_frag[2], d_frag[3]);  // cols 8..15

        // d_frag is the C-fragment of Y^T (Y = H16 @ X @ H16). One more free transpose
        // (swap regs 1/2) yields the B-fragment of Y, ready to store.
        tmp = d_frag[1];
        d_frag[1] = d_frag[2];
        d_frag[2] = tmp;

        // Store Y back to shared in-place (same column-major layout as the load above).
        sA[sA_base + group_id * 8 + tid_in_group] = d_frag[0];
        sA[sA_base + group_id * 8 + tid_in_group + 4] = d_frag[1];
        sA[sA_base + (group_id + 8) * 8 + tid_in_group] = d_frag[2];
        sA[sA_base + (group_id + 8) * 8 + tid_in_group + 4] = d_frag[3];
    }

    if constexpr (LOG_COLS > 8)
    {
        for (int level = 0; 4 * level < LOG_NCHUNK; ++level)
        {
            __syncthreads();

            const int base = 4 * level;
            const int cur_bits = (LOG_NCHUNK - base) < 4 ? (LOG_NCHUNK - base) : 4;
            const int dsize = 1 << cur_bits;
            const int total_tiles = (1 << (LOG_NCHUNK - cur_bits)) * 16;

            for (int tile = warp_id; tile < total_tiles; tile += WARPS_PER_ROW)
            {
                // 原本的 256 x (COLS/16/256) 划分为 16x16，每 16 个为一个 tile
                // 通过 tile 的编号可以计算出当前对应的原 chunk 和原 chunk 内的 offset
                const int r = tile >> 4;
                const int t = tile & 15;
                const int g = lane_id >> 2;
                const int tig = lane_id % 4;

                const int d_lo = 2 * tig;     // K rows 2*tig, 2*tig+1
                const int d_hi = 2 * tig + 8; // K rows 2*tig+8, 2*tig+9
                const int p_lo = g;           // N col 0..7  -> abs pos 16t + g
                const int p_hi = g + 8;       // N col 8..15 -> abs pos 16t + g + 8
                const int abs_lo = 16 * t + p_lo;
                const int abs_hi = 16 * t + p_hi;

                // B-fragment (16x16, K = digit, N = position): each b32 packs two
                // consecutive K rows (digit values), gathered across chunks.
                b32 b_frag[4];
                b_frag[0] = ((b32)stage_read(sA, base, cur_bits, dsize, d_lo,     r, abs_lo) & 0xFFFF) |
                            ((b32)stage_read(sA, base, cur_bits, dsize, d_lo + 1, r, abs_lo) << 16);
                b_frag[1] = ((b32)stage_read(sA, base, cur_bits, dsize, d_hi,     r, abs_lo) & 0xFFFF) |
                            ((b32)stage_read(sA, base, cur_bits, dsize, d_hi + 1, r, abs_lo) << 16);
                b_frag[2] = ((b32)stage_read(sA, base, cur_bits, dsize, d_lo,     r, abs_hi) & 0xFFFF) |
                            ((b32)stage_read(sA, base, cur_bits, dsize, d_lo + 1, r, abs_hi) << 16);
                b_frag[3] = ((b32)stage_read(sA, base, cur_bits, dsize, d_hi,     r, abs_hi) & 0xFFFF) |
                            ((b32)stage_read(sA, base, cur_bits, dsize, d_hi + 1, r, abs_hi) << 16);

                b32 r_frag[4];
                mma_m16_n8_k16<T>(h_frag[0], h_frag[1], h_frag[2], h_frag[3],
                                  b_frag[0], b_frag[1], r_frag[0], r_frag[1]);
                mma_m16_n8_k16<T>(h_frag[0], h_frag[1], h_frag[2], h_frag[3],
                                  b_frag[2], b_frag[3], r_frag[2], r_frag[3]);

                const int c_g = chunk_index(base, cur_bits, g, r);
                const int c_g8 = chunk_index(base, cur_bits, g + 8, r);
                if (g < dsize)     sA[c_g * 128 + (8 * t + tig)] = r_frag[0];
                if (g + 8 < dsize) sA[c_g8 * 128 + (8 * t + tig)] = r_frag[1];
                if (g < dsize)     sA[c_g * 128 + (8 * t + tig + 4)] = r_frag[2];
                if (g + 8 < dsize) sA[c_g8 * 128 + (8 * t + tig + 4)] = r_frag[3];
            }
        }

        __syncthreads();
    }

    b32 *out32 = reinterpret_cast<b32 *>(outputs);
    int out_base = (row * COLS + warp_id * ELEMENTS_PER_WARP) >> 1;
    #pragma unroll
    for (int i = 0; i < NUM_CHUNK; ++i)
    {
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            out32[out_base + i * 128 + lane_id * 4 + j] = sA[(warp_id * NUM_CHUNK + i) * 128 + lane_id * 4 + j];
        }
    }
}


// Host launcher: picks NUM_CHUNK so blockDim never exceeds 1024 threads
// (LOG_COLS 1..13 -> 1 chunk, 14 -> 2, 15 -> 4) and sizes dynamic shared memory
// to 2*COLS bytes on the tensor-core path (0 on the small scalar path).
template <typename T, int NUM_CHUNK, int LOG_COLS>
void launch_tc(const T *input, T *output, int rows, cudaStream_t stream)
{
    constexpr int COLS = 1 << LOG_COLS;
    constexpr int WARPS_PER_ROW = COLS / (256 * NUM_CHUNK);
    constexpr int BLOCK = (LOG_COLS < 8) ? WARP_SIZE : WARPS_PER_ROW * WARP_SIZE;
    constexpr int SMEM = (LOG_COLS < 8) ? 0 : WARPS_PER_ROW * NUM_CHUNK * 128 * 4;

    if constexpr (SMEM > 48 * 1024)
        cudaFuncSetAttribute(hadamard_tc_kernel<T, NUM_CHUNK, LOG_COLS>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);

    dim3 grid(rows), block(BLOCK);
    hadamard_tc_kernel<T, NUM_CHUNK, LOG_COLS><<<grid, block, SMEM, stream>>>(input, output, rows);
}

template <typename T>
void hadamard_tc(const T *input, T *output, int rows, int cols,
                 cudaStream_t stream = 0)
{
    static_assert(std::is_same<T, __half>::value ||
                      std::is_same<T, __nv_bfloat16>::value,
                  "T must be __half or __nv_bfloat16");

    int log_cols = 0;
    while ((1 << log_cols) < cols)
        ++log_cols;
    assert((1 << log_cols) == cols && "cols must be a power of 2");
    assert(log_cols >= 1 && log_cols <= 15 && "cols must be in [2, 32768]");

    switch (log_cols) {
        case 1:  launch_tc<T, 1, 1>(input, output, rows, stream); break;
        case 2:  launch_tc<T, 1, 2>(input, output, rows, stream); break;
        case 3:  launch_tc<T, 1, 3>(input, output, rows, stream); break;
        case 4:  launch_tc<T, 1, 4>(input, output, rows, stream); break;
        case 5:  launch_tc<T, 1, 5>(input, output, rows, stream); break;
        case 6:  launch_tc<T, 1, 6>(input, output, rows, stream); break;
        case 7:  launch_tc<T, 1, 7>(input, output, rows, stream); break;
        case 8:  launch_tc<T, 1, 8>(input, output, rows, stream); break;
        case 9:  launch_tc<T, 1, 9>(input, output, rows, stream); break;
        case 10: launch_tc<T, 1, 10>(input, output, rows, stream); break;
        case 11: launch_tc<T, 1, 11>(input, output, rows, stream); break;
        case 12: launch_tc<T, 1, 12>(input, output, rows, stream); break;
        case 13: launch_tc<T, 1, 13>(input, output, rows, stream); break;
        case 14: launch_tc<T, 2, 14>(input, output, rows, stream); break;
        case 15: launch_tc<T, 4, 15>(input, output, rows, stream); break;
    }
}

template void hadamard_tc<__half>(const __half*, __half*, int, int, cudaStream_t);
template void hadamard_tc<__nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, int, int, cudaStream_t);