#include "./utils.h"
#include <cstdint>

#define WARP_SIZE 32

// 4 warps per block
// Br = 64 (MMA_ATOM_M * NUM_WARP_IN_Q_BR * NUM_MMA_PER_WARP_Q_BR) = (16 * 4 * 1)
// Bc = 64 (MMA_ATOM_N * NUM_WARP_IN_K_BC * NUM_MMA_PER_WARP_K_BC) = (8 * 1 * 8)
// Q@KT (Br x HeadDim) @ (HeadDim x Bc)
// P@V (Br x Bc) @ (Bc x HeadDim)
template<
    const int MMA_ATOM_M,                       // 16
    const int MMA_ATOM_N,                       // 8
    const int MMA_ATOM_K,                       // 16
    const int NUM_WARP_IN_Q_BR,                 // 4
    const int NUM_WARP_IN_K_BC,                 // 1
    const int NUM_WARP_IN_P_BR,                 // 4
    const int NUM_WARP_IN_V_HEAD_DIM,           // 1
    const int NUM_MMA_PER_WARP_Q_BR,            // 1
    const int NUM_MMA_PER_WARP_K_BC,            // 8
    const int NUM_MMA_PER_WARP_P_BR,            // 1
    const int NUM_MMA_PER_WARP_V_HEAD_DIM,      // 8
    const int HEAD_DIM,
    const int NUM_THREADS
>
__global__ void flash_attention_fp16_spilt_q_shared_kv_kernel(
    const half* Q,
    const half* K,
    const half* V,
    half* O,
    const float scale,
    int batch_size, int target_seq_len, int src_seq_len,
    int query_heads, int kv_heads, bool is_causal)
{
    constexpr int Br = MMA_ATOM_M * NUM_WARP_IN_Q_BR * NUM_MMA_PER_WARP_Q_BR;
    constexpr int Bc = MMA_ATOM_N * NUM_WARP_IN_K_BC * NUM_MMA_PER_WARP_K_BC;

    const int Tc = div_ceil(src_seq_len, Bc);
    const int batch_id = blockIdx.x / query_heads;
    const int head_q_id = blockIdx.x % query_heads;
    const int head_group_num = query_heads / kv_heads;
    const int head_kv_id = head_q_id / head_group_num;
    const int tile_Br_id = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int warp_Q_id = warp_id;
    const int warp_KV_id = 0;

    __shared__ half smem_Q[Br][HEAD_DIM];
    __shared__ half smem_K[Bc][HEAD_DIM];
    __shared__ half smem_V[Bc][HEAD_DIM];

    uint32_t smem_Q_base_ptr = __cvta_generic_to_shared(smem_Q);
    uint32_t smem_K_base_ptr = __cvta_generic_to_shared(smem_K);
    uint32_t smem_V_base_ptr = __cvta_generic_to_shared(smem_V);

    // global Br row old max
    float reg_m_block_old[NUM_MMA_PER_WARP_Q_BR][2]; // in mma, each thread has 2 row in Nc
    float reg_l_block_old[NUM_MMA_PER_WARP_Q_BR][2];
    
    #pragma unroll
    for (int i = 0; i < NUM_MMA_PER_WARP_Q_BR; ++i) {
        reg_m_block_old[i][0] = -INFINITY;
        reg_m_block_old[i][1] = -INFINITY;
        reg_l_block_old[i][0] = 0.0f;
        reg_l_block_old[i][1] = 0.0f;
    }

    uint32_t reg_Q[NUM_MMA_PER_WARP_Q_BR][4];
    uint32_t reg_K[NUM_MMA_PER_WARP_K_BC][2];
    float reg_S[NUM_MMA_PER_WARP_Q_BR][NUM_MMA_PER_WARP_K_BC][4];
    uint32_t reg_P[NUM_MMA_PER_WARP_Q_BR][NUM_MMA_PER_WARP_K_BC][2];
    // Use fp32 for O accumulation to preserve precision across rescaling steps
    uint32_t reg_D[NUM_MMA_PER_WARP_P_BR][NUM_MMA_PER_WARP_V_HEAD_DIM][4];
    #pragma unroll
    for (int i = 0 ; i < NUM_MMA_PER_WARP_P_BR ; i ++){
        #pragma unroll
        for (int j = 0 ; j < NUM_MMA_PER_WARP_V_HEAD_DIM; ++j){
            reg_D[i][j][0] = 0;
            reg_D[i][j][1] = 0;
            reg_D[i][j][2] = 0;
            reg_D[i][j][3] = 0;
        }
    }

    // load Q from global memory to shared memory
    int load_smem_Q_Br = (tid / (NUM_THREADS / Br));
    int load_smem_Q_d = (tid % (NUM_THREADS / Br)) * (HEAD_DIM / (NUM_THREADS / Br));
    int load_gmem_Q_Br = tile_Br_id * Br + load_smem_Q_Br;
    int load_gmem_Q_d = load_smem_Q_d;
    if (load_gmem_Q_Br >= target_seq_len) {
        return;
    }

    // Load Q into shared memory
    // Q [batch_size, target_seq_len, query_heads, head_dim]
    const int load_num_Br_per_thread = HEAD_DIM / (NUM_THREADS / Br);
    int gmem_Q_offset = batch_id * target_seq_len * query_heads * HEAD_DIM +
        load_gmem_Q_Br * query_heads * HEAD_DIM +
        head_q_id * HEAD_DIM +
        load_gmem_Q_d;
    // HEAD_DIM Must be a multiple of 8
    #pragma unroll
    for (int i = 0; i < load_num_Br_per_thread; i += 8) {
        uint32_t smem_Q = smem_Q_base_ptr +
            (load_smem_Q_Br * HEAD_DIM + load_smem_Q_d + i) * sizeof(half);
        CP_ASYNC_CG(smem_Q, &Q[gmem_Q_offset + i], 16);
    }
    CP_ASYNC_COMMIT_GROUP();


    constexpr int VEC_ELEMS = 8;
    constexpr int KV_VECS_PER_ROW = HEAD_DIM / VEC_ELEMS;
    constexpr int KV_VECS_PER_TILE = Bc * KV_VECS_PER_ROW;
    #pragma unroll 1
    for (int tile_N_id = 0; tile_N_id < Tc; ++tile_N_id) {
        // first K
        if (tile_N_id == 0){
            #pragma unroll
            for (int vec_id = tid; vec_id < KV_VECS_PER_TILE; vec_id += NUM_THREADS) {
                int load_smem_K_Bc = vec_id / KV_VECS_PER_ROW;
                int load_smem_K_d = (vec_id % KV_VECS_PER_ROW) * VEC_ELEMS;
                int load_gmem_K_Bc_offset = tile_N_id * Bc + load_smem_K_Bc;
                int load_gmem_K_offset = batch_id * src_seq_len * kv_heads * HEAD_DIM +
                    load_gmem_K_Bc_offset * kv_heads * HEAD_DIM +
                    head_kv_id * HEAD_DIM + load_smem_K_d;
                uint32_t smem_K_now = smem_K_base_ptr +
                    (load_smem_K_Bc * HEAD_DIM + load_smem_K_d) * sizeof(half);
                CP_ASYNC_CG(smem_K_now, &K[load_gmem_K_offset], 16);
            }
            CP_ASYNC_COMMIT_GROUP();
            CP_ASYNC_WAIT_GROUP(0);
            __syncthreads();
        }
        // prefetch V
        #pragma unroll
        for (int vec_id = tid; vec_id < KV_VECS_PER_TILE; vec_id += NUM_THREADS) {
            int load_smem_V_Bc = vec_id / KV_VECS_PER_ROW;
            int load_smem_V_d = (vec_id % KV_VECS_PER_ROW) * VEC_ELEMS;
            int load_gmem_V_Bc_offset = tile_N_id * Bc + load_smem_V_Bc;
            int load_gmem_V_offset = batch_id * src_seq_len * kv_heads * HEAD_DIM +
                load_gmem_V_Bc_offset * kv_heads * HEAD_DIM +
                head_kv_id * HEAD_DIM + load_smem_V_d;
            uint32_t smem_V_now = smem_V_base_ptr +
                (load_smem_V_Bc * HEAD_DIM + load_smem_V_d) * sizeof(half);
            CP_ASYNC_CG(smem_V_now, &V[load_gmem_V_offset], 16);
        }
        CP_ASYNC_COMMIT_GROUP();

        const int tile_k_num = HEAD_DIM / MMA_ATOM_K;
        #pragma unroll
        for (int i = 0 ; i < NUM_MMA_PER_WARP_Q_BR; ++i) {
            #pragma unroll
            for (int j = 0; j < NUM_MMA_PER_WARP_K_BC; ++j) {
                reg_S[i][j][0] = 0.0f;
                reg_S[i][j][1] = 0.0f;
                reg_S[i][j][2] = 0.0f;
                reg_S[i][j][3] = 0.0f;
            }
        }

        for (int tile_k_id = 0; tile_k_id < tile_k_num; ++tile_k_id) {
            // load Q from shared memory to registers
            #pragma unroll
            for (int i = 0 ; i < NUM_MMA_PER_WARP_Q_BR; ++i) {
                // a warp load 16x16 matrix
                // each each thread has 4 elements
                int warp_smem_Q_Br = warp_Q_id * (NUM_MMA_PER_WARP_Q_BR * MMA_ATOM_M) + i * MMA_ATOM_M;
                int lane_smem_Q_Br = warp_smem_Q_Br + lane_id % 16;
                int lane_smem_Q_d = tile_k_id * MMA_ATOM_K + (lane_id / 16) * 8;
                uint32_t lane_smem_Q_ptr = smem_Q_base_ptr + (lane_smem_Q_Br * HEAD_DIM + lane_smem_Q_d) * sizeof(half);
                LDMATRIX_X4(reg_Q[i][0], reg_Q[i][1], reg_Q[i][2], reg_Q[i][3], lane_smem_Q_ptr);
            }
            // load K from shared memory to registers
            #pragma unroll
            for (int i = 0 ; i < NUM_MMA_PER_WARP_K_BC; ++i) {
                int warp_smem_K_Bc = warp_KV_id * (NUM_MMA_PER_WARP_K_BC * MMA_ATOM_N) + i * MMA_ATOM_N;
                int lane_smem_K_Bc = warp_smem_K_Bc + lane_id % 8;
                int lane_smem_K_d = tile_k_id * MMA_ATOM_K + ((lane_id / 8) % 2) * 8;
                uint32_t lane_smem_K_ptr = smem_K_base_ptr + (lane_smem_K_Bc * HEAD_DIM + lane_smem_K_d) * sizeof(half);
                LDMATRIX_X2(reg_K[i][0], reg_K[i][1], lane_smem_K_ptr);
            }
            #pragma unroll
            for (int j = 0; j < NUM_MMA_PER_WARP_K_BC; ++j) {
                HMMA16832(reg_S[0][j][0], reg_S[0][j][1], reg_S[0][j][2], reg_S[0][j][3],
                        reg_Q[0][0], reg_Q[0][1], reg_Q[0][2], reg_Q[0][3],
                        reg_K[j][0], reg_K[j][1],
                        reg_S[0][j][0], reg_S[0][j][1], reg_S[0][j][2], reg_S[0][j][3]);
            }
        }
        __syncthreads();

        // prefetch next K
        if (tile_N_id + 1 < Tc) {
            #pragma unroll
            for (int vec_id = tid; vec_id < KV_VECS_PER_TILE; vec_id += NUM_THREADS) {
                int load_smem_K_Bc = vec_id / KV_VECS_PER_ROW;
                int load_smem_K_d = (vec_id % KV_VECS_PER_ROW) * VEC_ELEMS;
                int load_gmem_K_Bc_offset = (tile_N_id + 1) * Bc + load_smem_K_Bc;
                int load_gmem_K_offset = batch_id * src_seq_len * kv_heads * HEAD_DIM +
                    load_gmem_K_Bc_offset * kv_heads * HEAD_DIM +
                    head_kv_id * HEAD_DIM + load_smem_K_d;
                uint32_t smem_K_now = smem_K_base_ptr +
                    (load_smem_K_Bc * HEAD_DIM + load_smem_K_d) * sizeof(half);
                CP_ASYNC_CG(smem_K_now, &K[load_gmem_K_offset], 16);
            }
            CP_ASYNC_COMMIT_GROUP();
        }

        float lane_row_m_new[NUM_MMA_PER_WARP_Q_BR][2];
        float lane_row_l_new[NUM_MMA_PER_WARP_Q_BR][2];
        #pragma unroll
        for (int i = 0; i < NUM_MMA_PER_WARP_Q_BR; ++i) {
            lane_row_m_new[i][0] = -INFINITY;
            lane_row_m_new[i][1] = -INFINITY;
            lane_row_l_new[i][0] = 0.0f;
            lane_row_l_new[i][1] = 0.0f;
        }
        #pragma unroll
        for (int j = 0 ; j < NUM_MMA_PER_WARP_K_BC ; j ++) {
            float* s4 = &reg_S[0][j][0];
            float now_max_0 = fmaxf(s4[0], s4[1]) * scale;
            float now_max_1 = fmaxf(s4[2], s4[3]) * scale;
            lane_row_m_new[0][0] = max(lane_row_m_new[0][0], now_max_0);
            lane_row_m_new[0][1] = max(lane_row_m_new[0][1], now_max_1);
        }
        lane_row_m_new[0][0] = warp_reduce_max<float, 4>(lane_row_m_new[0][0]);
        lane_row_m_new[0][1] = warp_reduce_max<float, 4>(lane_row_m_new[0][1]);
        
        // get row sum of P
        {
            float block_row_sum_new_0 = lane_row_m_new[0][0];
            float block_row_sum_new_1 = lane_row_m_new[0][1];
            float block_row_sum_old_0 = reg_m_block_old[0][0];
            float block_row_sum_old_1 = reg_m_block_old[0][1];
            // m new
            block_row_sum_new_0 = max(block_row_sum_new_0, block_row_sum_old_0);
            block_row_sum_new_1 = max(block_row_sum_new_1, block_row_sum_old_1);
            // rowsum of P
            #pragma unroll
            for (int i = 0 ; i < NUM_MMA_PER_WARP_K_BC ; i ++) {
                float* s4 = &reg_S[0][i][0];
                half* p4 = reinterpret_cast<half*>(&reg_P[0][i][0]);
                float4 row_p;
                row_p.x = __expf(
                    __fmaf_rn(s4[0], scale, -block_row_sum_new_0)
                );
                row_p.y = __expf(
                    __fmaf_rn(s4[1], scale, -block_row_sum_new_0)
                );
                row_p.z = __expf(
                    __fmaf_rn(s4[2], scale, -block_row_sum_new_1)
                );
                row_p.w = __expf(
                    __fmaf_rn(s4[3], scale, -block_row_sum_new_1)
                );
                lane_row_l_new[0][0] += (row_p.x + row_p.y);
                lane_row_l_new[0][1] += (row_p.z + row_p.w);
                // store P as fp16 for next MMA
                p4[0] = __float2half_rn(row_p.x);
                p4[1] = __float2half_rn(row_p.y);
                p4[2] = __float2half_rn(row_p.z);
                p4[3] = __float2half_rn(row_p.w);
            }
        }
        lane_row_l_new[0][0] = warp_reduce_sum<float, 4>(lane_row_l_new[0][0]);
        lane_row_l_new[0][1] = warp_reduce_sum<float, 4>(lane_row_l_new[0][1]);


        // wait
        if (tile_N_id + 1 < Tc) {
            CP_ASYNC_WAIT_GROUP(1);
        }else {
            // only V
            CP_ASYNC_WAIT_GROUP(0);
        }
        __syncthreads();


        uint32_t reg_V[NUM_MMA_PER_WARP_V_HEAD_DIM][2];
        float reg_O[NUM_MMA_PER_WARP_P_BR][NUM_MMA_PER_WARP_V_HEAD_DIM][4];

        #pragma unroll
        for (int i = 0 ; i < NUM_MMA_PER_WARP_P_BR ; ++i){
            #pragma unroll
            for (int j = 0 ; j < NUM_MMA_PER_WARP_V_HEAD_DIM ; j ++){
                reg_O[i][j][0] = 0.0f;
                reg_O[i][j][1] = 0.0f;
                reg_O[i][j][2] = 0.0f;
                reg_O[i][j][3] = 0.0f;
            }
        }
        
        // do P@V
        // (BrxBc)@(BcxHeadDim)
        // sum in Bc Dim
        const int tile_Bc_num = Bc / MMA_ATOM_K;
        #pragma unroll
        for (int tile_Bc_id = 0 ; tile_Bc_id < tile_Bc_num ; tile_Bc_id++) {
            // load V to reg
            #pragma unroll
            for (int i = 0 ; i < NUM_MMA_PER_WARP_V_HEAD_DIM ; i ++){
                int warp_smem_V_d = warp_KV_id * MMA_ATOM_N + i * MMA_ATOM_N;
                int lane_smem_V_Bc = tile_Bc_id * MMA_ATOM_K + lane_id % 16;
                int lane_smem_V_d = warp_smem_V_d;
                uint32_t lane_smem_V_ptr = smem_V_base_ptr + (lane_smem_V_Bc * HEAD_DIM + lane_smem_V_d) * sizeof(half);
                LDMATRIX_X2_T(reg_V[i][0], reg_V[i][1], lane_smem_V_ptr);
            }
            // do mma of P@V
            // because reg_P in warp is 16x8, and in mma the A should be 16x16
            // but reg_P has the NUM_WARP_PER_BC_K DIM
            int w = tile_Bc_id * 2;
            #pragma unroll
            for (int i = 0 ; i < NUM_MMA_PER_WARP_V_HEAD_DIM ; i ++) {
                HMMA16832(
                    reg_O[0][i][0], reg_O[0][i][1], reg_O[0][i][2], reg_O[0][i][3],
                    reg_P[0][w][0], reg_P[0][w][1], reg_P[0][w+1][0], reg_P[0][w+1][1],
                    reg_V[i][0], reg_V[i][1],
                    reg_O[0][i][0], reg_O[0][i][1], reg_O[0][i][2], reg_O[0][i][3]
                );
            }
        }
        __syncthreads();

        // get l and O
        {
            float block_row_max_new_0 = lane_row_m_new[0][0];
            float block_row_max_new_1 = lane_row_m_new[0][1];
            float block_row_sum_new_0 = lane_row_l_new[0][0];
            float block_row_sum_new_1 = lane_row_l_new[0][1];

            float block_row_max_old_0 = reg_m_block_old[0][0];
            float block_row_max_old_1 = reg_m_block_old[0][1];
            block_row_max_new_0 = max(block_row_max_new_0, block_row_max_old_0);
            block_row_max_new_1 = max(block_row_max_new_1, block_row_max_old_1);
            block_row_max_old_0 = (
                tile_N_id > 0 ? block_row_max_old_0 : block_row_max_new_0
            );
            block_row_max_old_1 = (
                tile_N_id > 0 ? block_row_max_old_1 : block_row_max_new_1
            );
            float alpha0 = __expf(block_row_max_old_0 - block_row_max_new_0);
            float alpha1 = __expf(block_row_max_old_1 - block_row_max_new_1);
            #pragma unroll
            for (int i = 0 ; i < NUM_MMA_PER_WARP_V_HEAD_DIM ; i ++) {
                float* o4 = &reg_O[0][i][0];
                float* d4 = reinterpret_cast<float*>(&reg_D[0][i][0]);
                d4[0] = __fmaf_rn(alpha0, d4[0], o4[0]);
                d4[1] = __fmaf_rn(alpha0, d4[1], o4[1]);
                d4[2] = __fmaf_rn(alpha1, d4[2], o4[2]);
                d4[3] = __fmaf_rn(alpha1, d4[3], o4[3]);
            }
            float block_row_sum_old_0 = reg_l_block_old[0][0];
            float block_row_sum_old_1 = reg_l_block_old[0][1];
            reg_l_block_old[0][0] = __fmaf_rn(
                alpha0, block_row_sum_old_0, block_row_sum_new_0
            );
            reg_l_block_old[0][1] = __fmaf_rn(
                alpha1, block_row_sum_old_1, block_row_sum_new_1
            );
            reg_m_block_old[0][0] = block_row_max_new_0;
            reg_m_block_old[0][1] = block_row_max_new_1;
        }

        if (tile_N_id + 1 < Tc) {
            CP_ASYNC_WAIT_GROUP(0);
            __syncthreads();
        }
    }

    // Update Final O
    {
        float rl0 = __frcp_rn(reg_l_block_old[0][0]);
        float rl1 = __frcp_rn(reg_l_block_old[0][1]);
        #pragma unroll
        for (int i = 0 ; i < NUM_MMA_PER_WARP_V_HEAD_DIM ; i ++) {
            float* d4f = reinterpret_cast<float*>(&reg_D[0][i][0]);
            half* d4h = reinterpret_cast<half*>(&reg_D[0][i][0]);
            d4h[0] = __float2half_rn(rl0 * d4f[0]);
            d4h[1] = __float2half_rn(rl0 * d4f[1]);
            d4h[2] = __float2half_rn(rl1 * d4f[2]);
            d4h[3] = __float2half_rn(rl1 * d4f[3]);
        }
    }

    // Store O (Br, D) to Global
    {
        for (int i = 0 ; i < NUM_MMA_PER_WARP_V_HEAD_DIM ; ++i) {
            uint32_t reg_Z[2][4];
            reg_Z[0][0] = reg_D[0][i][0];
            reg_Z[1][0] = reg_D[0][i][1];
            reg_Z[0][1] = __shfl_sync(0xffffffff, reg_Z[0][0], lane_id + 1, 4);
            reg_Z[0][2] = __shfl_sync(0xffffffff, reg_Z[0][0], lane_id + 2, 4);
            reg_Z[0][3] = __shfl_sync(0xffffffff, reg_Z[0][0], lane_id + 3, 4);
            reg_Z[1][1] = __shfl_sync(0xffffffff, reg_Z[1][0], lane_id + 1, 4);
            reg_Z[1][2] = __shfl_sync(0xffffffff, reg_Z[1][0], lane_id + 2, 4);
            reg_Z[1][3] = __shfl_sync(0xffffffff, reg_Z[1][0], lane_id + 3, 4);
            int gmem_O_base = batch_id * target_seq_len * query_heads * HEAD_DIM;
            if (lane_id % 4 == 0) {
                int warp_O_Br = warp_Q_id * MMA_ATOM_M;
                int gmem_warp_O_Br = tile_Br_id * Br + warp_O_Br + lane_id / 4;
                int warp_O_d = warp_KV_id * MMA_ATOM_N * NUM_MMA_PER_WARP_V_HEAD_DIM + i * MMA_ATOM_N;
                int gmem_warp_O_d = warp_O_d;
                int gmem_O_row_0 = gmem_warp_O_Br;
                int gmem_O_row_1 = gmem_warp_O_Br + 8;
                int gmem_O_addr_0 = gmem_O_base + gmem_O_row_0 * query_heads * HEAD_DIM + head_q_id * HEAD_DIM + gmem_warp_O_d;
                int gmem_O_addr_1 = gmem_O_base + gmem_O_row_1 * query_heads * HEAD_DIM + head_q_id * HEAD_DIM + gmem_warp_O_d;
                LDST128BITS(O[gmem_O_addr_0]) = LDST128BITS(reg_Z[0][0]);
                LDST128BITS(O[gmem_O_addr_1]) = LDST128BITS(reg_Z[1][0]);
            }
        }
        
    }
}

