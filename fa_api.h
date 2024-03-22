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


#if defined(_WIN32)
EXPORT void fa_forward(void* q, void* k, void* v, void* qkv, void* softmax_lse,
    int head_dim, int seqlen_q, int seqlen_k, int num_heads, int num_heads_k, float scale, cudaStream_t stream);

#else
// EXPORT const char* test();
#endif
#endif