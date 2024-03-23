#ifndef __FA_API__
#define __FA_API__

#include <cutlass/numeric_types.h>
#include <stdio.h>
#include "flash.h"
#include "static_switch.h"

#if defined(_WIN32)
#ifdef CXX_BUILD
#define EXPORT __declspec(dllexport) 
#else
#define EXPORT __declspec(dllimport) 
#endif
#else
#define EXPORT
#endif

EXPORT void flash_attn_fwd(void* q, void* k, void* v, void* attn_bias, void* qkv, void* softmax_lse,
    const int head_dim, const int seqlen_q, const int seqlen_k, const int num_heads, const int num_heads_kv,
    const int attn_bias_heads, const int batch_size, const float scale, cudaStream_t stream);
#endif