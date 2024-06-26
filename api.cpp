#include <cuda_runtime.h>
#if !defined(_WIN32)
#include <stddef.h>
#endif
#include "fa_api.h"

#ifndef M_LOG2E
#define M_LOG2E    1.44269504088896340736
#endif

void set_params_fprop(Flash_fwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,

                      // device pointers
                     void* q,
                     void* k,
                     void* v,
                     void* qkv,
                      void *softmax_lse_d,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *seqused_k,
                      void *p_d,

                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      bool seqlenq_ngroups_swapped=false) {

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.is_bf16 = false;
    params.q_ptr = q;
    params.k_ptr = k;
    params.v_ptr = v;

    params.o_ptr = qkv;

    if (cu_seqlens_q_d == nullptr) {
        if (seqlenq_ngroups_swapped) {
             params.q_batch_stride *= seqlen_q;
             params.o_batch_stride *= seqlen_q;
        }
    }

    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
    params.seqused_k = static_cast<int *>(seqused_k);

    // P = softmax(QK^T)
    params.p_ptr = p_d;

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;

    // Set the different scale values.
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;

    // Causal is the special case where window_size_right == 0 and window_size_left < 0.
    // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
    params.is_causal = window_size_left < 0 && window_size_right == 0;

    if (window_size_left < 0 && window_size_right >= 0) { window_size_left = seqlen_k; }
    if (window_size_left >= 0 && window_size_right < 0) { window_size_right = seqlen_k; }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    // #ifdef FLASHATTENTION_DISABLE_LOCAL
    //     TORCH_CHECK(params.is_causal || (window_size_left < 0 && window_size_right < 0),
    //         "This flash attention build does not support local attention.");
    // #endif

    params.is_seqlens_k_cumulative = true;
}


// Find the number of splits that maximizes the occupancy. For example, if we have
// batch * n_heads = 48 and we have 108 SMs, having 2 splits (efficiency = 0.89) is
// better than having 3 splits (efficiency = 0.67). However, we also don't want too many
// splits as that would incur more HBM reads/writes.
// So we find the best efficiency, then find the smallest number of splits that gets 85%
// of the best efficiency.
inline int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits) {
    // If we have enough to almost fill the SMs, then just use 1 split
    if (batch_nheads_mblocks >= 0.8f * num_SMs) { return 1; }
    max_splits = std::min({max_splits, num_SMs, num_n_blocks});
    float max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
    // Some splits are not eligible. For example, if we have 64 blocks and choose 11 splits,
    // we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have 6 * 11 + (-2) blocks
    // (i.e. it's 11 splits anyway).
    // So we check if the number of blocks per split is the same as the previous num_splits.
    auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
        return num_splits == 1 || ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
    };
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) {
            efficiency.push_back(0.f);
        } else {
            float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
            float eff = n_waves / ceil(n_waves);
            // printf("num_splits = %d, eff = %f\n", num_splits, eff);
            if (eff > max_efficiency) { max_efficiency = eff; }
            efficiency.push_back(eff);
        }
    }
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) { continue; }
        if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
            // printf("num_splits chosen = %d\n", num_splits);
            return num_splits;
        }
    }
    return 1;
}

void set_params_splitkv(Flash_fwd_params &params, const int batch_size,
    const int num_heads, const int head_size, const int max_seqlen_k, const int max_seqlen_q,
    const int head_size_rounded, const float p_dropout,
    const int num_splits) {

    cudaDeviceProp dprops;
    cudaGetDeviceProperties(&dprops, 0);

    // This needs to match with run_mha_fwd_splitkv_dispatch
    const int block_n = head_size <= 64 ? 256 : (head_size <= 128 ? 128 : 64);
    const int num_n_blocks = (max_seqlen_k + block_n - 1) / block_n;

    // Technically kBlockM = 64 only for the splitKV kernels, not the standard kernel.
    // In any case we don't expect seqlen_q to be larger than 64 for inference.
    const int num_m_blocks = (max_seqlen_q + 64 - 1) / 64;
    params.num_splits = num_splits;
    if (p_dropout == 0.0f) {  // SplitKV is not implemented for dropout
        if (num_splits < 1) {
            params.num_splits = num_splits_heuristic(batch_size * num_heads * num_m_blocks, dprops.multiProcessorCount, num_n_blocks, 128);
        }
        // TORCH_CHECK(params.num_splits <= 128, "num_splits > 128 not supported");
    }
}

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel=false) {
    HEADDIM_SWITCH(params.d, [&] {
        if (params.num_splits <= 1 && !force_split_kernel) {  // If we don't set it num_splits == 0
            run_mha_fwd_<cutlass::half_t, kHeadDim>(params, stream);
        } else {
            run_mha_fwd_splitkv_dispatch<cutlass::half_t, kHeadDim>(params, stream);
        }
    });
}

void flash_attn_fwd(void* q, void* k, void* v, void* attn_bias, void* qkv, void* softmax_lse,
    const int head_dim, const int seqlen_q, const int seqlen_k, const int num_heads, const int num_heads_kv,
    const int attn_bias_heads, const int batch_size, const float scale, cudaStream_t stream) {

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size = round_multiple(head_dim, 8);
    const int head_size_rounded = round_multiple(head_size, 32);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    Flash_fwd_params params;

    set_params_fprop(params,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_kv,
                     head_size, head_size_rounded,
                     q, k, v, qkv,
                     softmax_lse,
                     /*cu_seqlens_q_d=*/nullptr,
                     /*cu_seqlens_k_d=*/nullptr,
                     /*seqused_k=*/nullptr,
                     nullptr,
                     0.0f,
                     scale,
                     -1,
                     -1);

    set_params_splitkv(params, batch_size, num_heads,
                       head_size, seqlen_k, seqlen_q,
                       head_size_rounded, 0.0f, /*num_splits*/0);
    
    if (params.num_splits > 1) {
        cudaMallocAsync((void**)&params.softmax_lseaccum_ptr, params.num_splits * batch_size * num_heads * seqlen_k * sizeof(float), stream);
        cudaMallocAsync((void**)&params.oaccum_ptr, params.num_splits * batch_size * num_heads * seqlen_q * head_size_rounded * sizeof(float), stream);
    }

    // All stride are in elements, not bytes.
    params.q_row_stride = num_heads * head_dim;
    params.k_row_stride = num_heads_kv * head_dim;
    params.v_row_stride = num_heads_kv * head_dim;

    params.q_head_stride = head_dim;
    params.k_head_stride = head_dim;
    params.v_head_stride = head_dim;

    params.o_row_stride = num_heads * head_dim;
    params.o_head_stride = head_dim;

    params.q_batch_stride = num_heads * head_dim * seqlen_q;
    params.k_batch_stride = num_heads_kv * head_dim * seqlen_k;
    params.v_batch_stride = num_heads_kv * head_dim * seqlen_k;
    params.o_batch_stride = num_heads * head_dim * seqlen_q;

    params.attn_bias_batch_stride = 0;
    if(attn_bias) {
        params.attn_bias_ptr =          attn_bias;
        params.attn_bias_head_stride =  seqlen_k * seqlen_q;
        params.attn_bias_q_stride =     seqlen_k;
    } else {
        params.attn_bias_ptr = attn_bias;
        params.attn_bias_head_stride = 0;
        params.attn_bias_q_stride    = 0;
    }

    if ((attn_bias_heads == 1) && (num_heads != 1)) {
        params.attn_bias_head_stride = 0;
    }

    run_mha_fwd(params, stream);

    if (params.num_splits > 1) {
        cudaFreeAsync(params.softmax_lseaccum_ptr, stream);
        cudaFreeAsync(params.oaccum_ptr, stream);
    }
}
