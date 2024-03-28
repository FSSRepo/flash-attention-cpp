#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <algorithm>
#include "flash_llama.h"
#include "fa_api.h"
#include "main.h"
#include "util.h"

#define PADD(x, n) (((x) + (n) - 1) & ~((n) - 1))

void cap_test() {
    bool flash_attn_orig = true;
    int head_dim = 128, batch_size = 1, num_heads = 32, kv_size = 256, num_kv_heads = 32;
    const int r_kv_heads = num_heads / num_kv_heads;

    float scale = 1.0f / sqrtf((float)head_dim);

    // input buffers
    float* query =  (float*)malloc(head_dim * batch_size * num_heads * sizeof(float)); // assume batch size 1
    float* key =    (float*)malloc(head_dim * kv_size * num_kv_heads * sizeof(float));
    float* value =  (float*)malloc(head_dim * kv_size * num_kv_heads * sizeof(float));
    float* mask =   (float*)malloc(kv_size * batch_size * sizeof(float));

    // output buffers
    float* qkv =            (float*)malloc(head_dim  * batch_size  * num_heads * sizeof(float)); // assume batch size 1
    half* qkv_cuda =        (half*)malloc(head_dim  * batch_size * num_heads * sizeof(half)); // assume batch size 1
    float* qkv_cuda_f32 =   (float*)malloc(head_dim  * batch_size  * num_heads * sizeof(float)); // assume batch size 1
    float* scores =         (float*)malloc(kv_size * batch_size * num_heads * sizeof(float)); // QK^T

    // fill buffers
    fill_buffer(qkv, 0.0f, head_dim * batch_size * num_heads);
    fill_buffer(scores, 0.0f, kv_size * batch_size * num_heads);

    random(query, head_dim * batch_size * num_heads);
    random(key,   head_dim * kv_size * num_kv_heads);
    random(value, head_dim * kv_size * num_kv_heads);
    random(mask,  kv_size * batch_size);

    if(true) {
        // cpu cmputation
        for(int h = 0; h < num_heads; h++) {
            mulmat_cpu(query + h*batch_size*head_dim, key + ((h/r_kv_heads) * head_dim*kv_size), mask, scores + h*batch_size*kv_size, batch_size, kv_size, head_dim, scale, true);
            softmax(scores + h*batch_size*kv_size, kv_size, batch_size, h);
        }

        // printf("softmax(QKT)\n");

        // print_array("Scores", scores, kv_size, 8);

        for(int h = 0; h < num_heads; h++) {
            mulmat_cpu(scores + h*batch_size*kv_size, value + ((h/r_kv_heads) * head_dim*kv_size), nullptr, qkv + h*batch_size*head_dim, batch_size, head_dim, kv_size, 1.0f);
        }

        print_array("Reference", qkv, 1, 16, head_dim);

        fill_buffer(qkv_cuda, 0.0f, head_dim * batch_size * num_heads);
    }

    if(true) {
        // cuda cumputation
        half * query_f16 =   (half*)malloc(head_dim * batch_size * num_heads * sizeof(half));
        half * key_f16 =     (half*)malloc(head_dim * kv_size * num_kv_heads * sizeof(half));
        half * value_f16 =   (half*)malloc(head_dim * kv_size * num_kv_heads * sizeof(half));

        // half * value_f16_nT =   (half*)malloc(head_dim * kv_size * num_kv_heads * sizeof(half));
        half * mask_f16_padded = (half*)malloc(kv_size * PADD(batch_size, 32) * sizeof(half));

        for(int b = 0; b < PADD(batch_size, 32); b ++) {
            for(int i = 0; i < kv_size; i ++) {
                if(b < batch_size) {
                    mask_f16_padded[b*kv_size + i] = __float2half(mask[b*kv_size + i]);
                } else {
                    mask_f16_padded[b*kv_size + i] =  __float2half(0.0f);
                }
            }
        }

//         for(int i = 0; i < head_dim * kv_size * num_kv_heads; i ++) {
//             key_f16[i] = __float2half(key[i]);
// #ifndef FA_KV_BLOCK_256
//             value_f16[i] = __float2half(value[i]);
// #else
//             value_f16_nT[i] = __float2half(value[i]);
// #endif
//         }

        // head_dim x kv_size x num_heads => head_dim x num_heads x kv_size

        if(flash_attn_orig) {
            for(int h = 0; h < num_kv_heads; h++) {
                for(int c = 0; c < head_dim; c++) {
                    for(int r = 0; r < kv_size; r++) {
                        key_f16  [r*num_kv_heads*head_dim + h*head_dim + c] = __float2half(key  [h*kv_size*head_dim + r*head_dim + c]);
                        value_f16[r*num_kv_heads*head_dim + h*head_dim + c] = __float2half(value[h*kv_size*head_dim + r*head_dim + c]);
                    }
                }
            }

            for(int h = 0; h < num_heads; h++) {
                for(int c = 0; c < head_dim; c++) {
                    for(int r = 0; r < batch_size; r++) {
                        query_f16[r*num_heads*head_dim + h*head_dim + c] = __float2half(query[h*batch_size*head_dim + r*head_dim + c]);
                    }
                }
            }
        } else {
            for(int i = 0; i < head_dim * kv_size * num_kv_heads; i ++) {
                key_f16[i] = __float2half(key[i]);
                value_f16[i] = __float2half(value[i]);
            }
        }

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        half  *d_query, *d_qkv, *d_key, *d_value, *d_mask;
        float *d_softmax_lse, * d_query_f32, * d_qkv_f32;

        cudaMalloc((void **)&d_query,  head_dim  * batch_size * num_heads * sizeof(half));
        cudaMalloc((void **)&d_query_f32,  head_dim  * batch_size * num_heads * sizeof(float));
        cudaMalloc((void **)&d_qkv_f32,  head_dim  * batch_size * num_heads * sizeof(float));

        cudaMalloc((void **)&d_key,     head_dim * kv_size * num_kv_heads * sizeof(half));
        cudaMalloc((void **)&d_value,   head_dim * kv_size * num_kv_heads * sizeof(half));
        cudaMalloc((void **)&d_mask,  kv_size * PADD(batch_size, 32) * sizeof(half));

        // cudaMalloc((void **)&d_value_nT,  head_dim * kv_size * num_kv_heads * sizeof(half));
        // cudaMalloc((void **)&d_mask,    kv_size * sizeof(half));
        // cudaMalloc((void **)&d_padded_mask,  32 *  kv_size * sizeof(half));

        cudaMalloc((void **)&d_qkv,     head_dim  * batch_size * num_heads * sizeof(half));

        // copy CPU data to GPU memory blocks
        cudaMemcpyAsync(d_query, query_f16, head_dim * batch_size * num_heads * sizeof(half), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_query_f32, query, head_dim * batch_size * num_heads * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_key,   key_f16,   head_dim * kv_size * num_kv_heads * sizeof(half), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_value, value_f16, head_dim * kv_size * num_kv_heads * sizeof(half), cudaMemcpyHostToDevice, stream);
        
        // cudaMemcpyAsync(d_value_nT, value_f16_nT, head_dim * kv_size * num_kv_heads * sizeof(half), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_mask,  mask_f16_padded,  kv_size * PADD(batch_size, 32) * sizeof(half), cudaMemcpyHostToDevice, stream);
        // cudaMemcpyAsync(d_padded_mask,  mask_f16_padded, 32 * kv_size * sizeof(half), cudaMemcpyHostToDevice, stream);

        cudaStreamSynchronize(stream);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, stream);
        cudaMalloc((void **)&d_softmax_lse,  num_heads * batch_size * sizeof(float));
        if(flash_attn_orig) {
            flash_attn_fwd(d_query, d_key, d_value, d_mask, d_qkv, d_softmax_lse, head_dim, batch_size, kv_size, num_heads, num_kv_heads, 1, 1, scale, stream);
        } else {
            constexpr int nqpb = 16;
            constexpr int ncpw = 128;

            const int nwarps = batch_size <= nqpb ? std::max(2, std::min((int) kv_size/ncpw, 8)) : 1;

            dim3 blocks_num((batch_size + nqpb - 1) / nqpb, num_heads, 1);
            dim3 block_dim(32, nwarps, 1);

            const size_t shmem_f_ = nqpb*(head_dim + nwarps*(ncpw + nqpb))*(sizeof(float)/2);

            flash_attn_ext_f16<64, nqpb, ncpw><<<blocks_num, block_dim, shmem_f_, stream>>>(
                (const char*)d_query_f32, (const char*)d_key, (const char*)d_value, (const char*)d_mask, d_qkv_f32, scale,
                head_dim, batch_size, num_heads, 1, // query
                head_dim, kv_size, num_kv_heads, 1, // key value
                PADD(batch_size, 32), kv_size * 2, // masks
                head_dim * 4, head_dim * batch_size * 4, head_dim * batch_size * num_heads * 4, // nb query
                head_dim * 2, head_dim * kv_size * 2, head_dim  * kv_size * num_kv_heads * 2, // nb key value
                // head_dim * 2, head_dim * kv_size * 2, head_dim * kv_size * num_heads * 2, // nb key value
                head_dim, num_heads, batch_size, 1);
        }
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);

        float millis = 0.0f;
        cudaEventElapsedTime(&millis, start, stop);

        // transfer data from device to host
        if(flash_attn_orig) {
            cudaMemcpyAsync(qkv_cuda, d_qkv, head_dim * batch_size * num_heads * sizeof(half), cudaMemcpyDeviceToHost, stream);
        } else {
            cudaMemcpyAsync(qkv_cuda_f32, d_qkv_f32, head_dim * batch_size * num_heads * sizeof(float), cudaMemcpyDeviceToHost, stream);
        }
        // cudaMemcpyAsync(key_f16, d_key, head_dim * kv_size * num_heads * sizeof(half), cudaMemcpyDeviceToHost, stream);

        cudaStreamSynchronize(stream);

        float max_diff = 0.0f;
        int head_idx = 0, batch_idx = 0, dim_idx = 0;

        if(flash_attn_orig) {
            print_array("Flash Attention", qkv_cuda, 1, 16, head_dim);
            for(int h = 0; h < num_heads; h++) {
                for(int b = 0; b < batch_size; b++) {
                    for(int i = 0; i < head_dim; i++) {
                        if(fabs(qkv[h*head_dim*batch_size + b*head_dim + i] - __half2float(qkv_cuda[b*num_heads*head_dim + h*head_dim + i]) ) > max_diff) {
                            max_diff = fabs(qkv[h*head_dim*batch_size + b*head_dim + i] -  __half2float(qkv_cuda[b*num_heads*head_dim + h*head_dim + i]));
                            head_idx = h;
                            batch_idx = b;
                            dim_idx = i;
                        }
                    }
                }
            }
            printf("\ncuda time: %.4f ms\n", millis);

            printf("R (%.4f) CUDA(%.4f) diff: %.4f - head = %d, batch = %d, dim = %d\n",
                qkv[head_idx*head_dim*batch_size + batch_idx*head_dim + dim_idx],  __half2float(qkv_cuda[batch_idx*num_heads*head_dim +  head_idx*head_dim + dim_idx]), max_diff, head_idx, batch_idx, dim_idx);
        } else {
            print_array("Flash Attention Llama", qkv_cuda_f32, 1, 16, head_dim);
            for(int h = 0; h < num_heads; h++) {
                for(int b = 0; b < batch_size; b++) {
                    for(int i = 0; i < head_dim; i++) {
                        if(fabs(qkv[h*head_dim*batch_size + b*head_dim + i] - qkv_cuda_f32[b*num_heads*head_dim + h*head_dim + i] ) > max_diff) {
                            max_diff = fabs(qkv[h*head_dim*batch_size + b*head_dim + i] - qkv_cuda_f32[b*num_heads*head_dim + h*head_dim + i]);
                            head_idx = h;
                            batch_idx = b;
                            dim_idx = i;
                        }
                    }
                }
            }

            printf("\ncuda time: %.4f ms\n", millis);
            
            // printf("REFERENCE: \n");
            // for(int r = 0;r < 8; r ++) {
            //     printf("%0.5ff, ", qkv[31*head_dim*batch_size + 0*head_dim + r]);
            // }
            // printf("\n");

            // printf("CUDA: \n");
            //  for(int r = 0;r < 8; r ++) {
            //     printf("%0.5ff, ", qkv_cuda_f32[0*num_heads*head_dim + 31*head_dim + r]);
            // }
            // printf("\n");

            printf("R (%.4f) CUDA(%.4f) diff: %.4f - head = %d, batch = %d, dim = %d\n",
                qkv[head_idx*head_dim*batch_size + batch_idx*head_dim + dim_idx],  qkv_cuda_f32[batch_idx*num_heads*head_dim +  head_idx*head_dim + dim_idx], max_diff, head_idx, batch_idx, dim_idx);

        }

        // clean up device memory
        cudaFree(d_query);
        cudaFree(d_key);
        cudaFree(d_value);
        cudaFree(d_qkv);
    }

    free(query);
    free(key);
    free(value);
    free(qkv);
    free(qkv_cuda);
    free(scores);
}

void real_test() {
    bool flash_attn_orig = true;

    tensor* tensor_q =              load_tensor_from_file("C:\\proyectos\\kernel-data\\tg\\q-256.tensor");
    tensor* tensor_k =              load_tensor_from_file("C:\\proyectos\\kernel-data\\tg\\k-256.tensor");
    tensor* tensor_v =              load_tensor_from_file("C:\\proyectos\\kernel-data\\tg\\v-256.tensor");
    tensor* tensor_mask =           load_tensor_from_file("C:\\proyectos\\kernel-data\\tg\\mask-256.tensor");
    tensor* tensor_qkv_ref =        load_tensor_from_file("C:\\proyectos\\kernel-data\\tg\\qkv-256.tensor");

    int head_dim = 64, batch_size = 1, num_heads = 32, kv_size = 256, num_kv_heads = 32;
    const int r_kv_heads = num_heads / num_kv_heads;

    float scale = 1.0f / sqrtf((float)head_dim);

    // input buffers
    float* query =  (float*)malloc(head_dim * batch_size * num_heads * sizeof(float)); // assume batch size 1
    float* key =    (float*)malloc(head_dim * kv_size * num_kv_heads * sizeof(float));
    float* value =  (float*)malloc(head_dim * kv_size * num_kv_heads * sizeof(float));
    float* mask =   (float*)malloc(kv_size * batch_size * sizeof(float));

    // output buffers
    float* qkv =           (float*) tensor_qkv_ref->data; // assume batch size 1
    half* qkv_cuda =        (half*)malloc(head_dim  * batch_size * num_heads * sizeof(half)); // assume batch size 1
    float* qkv_cuda_f32 =   (float*)malloc(head_dim  * batch_size  * num_heads * sizeof(float)); // assume batch size 1

    print_array("Reference", qkv, 1, 16, head_dim);

    // fill buffers
    // fill_buffer(qkv, 0.0f, head_dim * batch_size * num_heads);

    fill_buffer(qkv_cuda, 0.0f, head_dim * batch_size * num_heads);

    if(true) {
        // cuda cumputation
        half * query_f16 =   (half*)malloc(head_dim * batch_size * num_heads * sizeof(half));
        half * key_f16 =     (half*)malloc(head_dim * kv_size * num_kv_heads * sizeof(half));
        half * value_f16 =   (half*)malloc(head_dim * kv_size * num_kv_heads * sizeof(half));

        // half * value_f16_nT =   (half*)malloc(head_dim * kv_size * num_kv_heads * sizeof(half));
        half * mask_f16_padded = (half*)malloc(kv_size * PADD(batch_size, 32) * sizeof(half));

        for(int b = 0; b < PADD(batch_size, 32); b ++) {
            for(int i = 0; i < kv_size; i ++) {
                if(b < batch_size) {
                    mask_f16_padded[b*kv_size + i] = __float2half(mask[b*kv_size + i]);
                } else {
                    mask_f16_padded[b*kv_size + i] =  __float2half(0.0f);
                }
            }
        }

//         for(int i = 0; i < head_dim * kv_size * num_kv_heads; i ++) {
//             key_f16[i] = __float2half(key[i]);
// #ifndef FA_KV_BLOCK_256
//             value_f16[i] = __float2half(value[i]);
// #else
//             value_f16_nT[i] = __float2half(value[i]);
// #endif
//         }

        // head_dim x kv_size x num_heads => head_dim x num_heads x kv_size

        if(flash_attn_orig) {
            // for(int h = 0; h < num_kv_heads; h++) {
            //     for(int c = 0; c < head_dim; c++) {
            //         for(int r = 0; r < kv_size; r++) {
            //             key_f16  [r*num_kv_heads*head_dim + h*head_dim + c] = __float2half(key  [h*kv_size*head_dim + r*head_dim + c]);
            //             value_f16[r*num_kv_heads*head_dim + h*head_dim + c] = __float2half(value[h*kv_size*head_dim + r*head_dim + c]);
            //         }
            //     }
            // }

            for(int h = 0; h < num_heads; h++) {
                for(int c = 0; c < head_dim; c++) {
                    for(int b = 0; b < batch_size; b++) {
                        query_f16[b*num_heads*head_dim + h*head_dim + c] = __float2half(((float*)tensor_q->data)[b*num_heads*head_dim + h*head_dim + c]);
                    }
                }
            }
        } else {
            for(int i = 0; i < head_dim * kv_size * num_kv_heads; i ++) {
                key_f16[i] = __float2half(key[i]);
                value_f16[i] = __float2half(value[i]);
            }
        }

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        half  *d_query, *d_qkv, *d_key, *d_value, *d_mask;
        float *d_softmax_lse, * d_query_f32, * d_qkv_f32;

        cudaMalloc((void **)&d_query,  head_dim  * batch_size * num_heads * sizeof(half));
        cudaMalloc((void **)&d_query_f32,  head_dim  * batch_size * num_heads * sizeof(float));
        cudaMalloc((void **)&d_qkv_f32,  head_dim  * batch_size * num_heads * sizeof(float));

        cudaMalloc((void **)&d_key,     head_dim * kv_size * num_kv_heads * sizeof(half));
        cudaMalloc((void **)&d_value,   head_dim * kv_size * num_kv_heads * sizeof(half));
        cudaMalloc((void **)&d_mask,    kv_size * PADD(batch_size, 32) * sizeof(half));

        // cudaMalloc((void **)&d_value_nT,  head_dim * kv_size * num_kv_heads * sizeof(half));
        // cudaMalloc((void **)&d_mask,    kv_size * sizeof(half));
        // cudaMalloc((void **)&d_padded_mask,  32 *  kv_size * sizeof(half));

        cudaMalloc((void **)&d_qkv,     head_dim  * batch_size * num_heads * sizeof(half));

        // copy CPU data to GPU memory blocks
        cudaMemcpyAsync(d_query,     query_f16,      head_dim * batch_size * num_heads * sizeof(half), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_query_f32, tensor_q->data, head_dim * batch_size * num_heads * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_key,       tensor_k->data, head_dim * kv_size * num_kv_heads * sizeof(half), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_value,     tensor_v->data, head_dim * kv_size * num_kv_heads * sizeof(half), cudaMemcpyHostToDevice, stream);

        // cudaMemcpyAsync(d_value_nT, value_f16_nT, head_dim * kv_size * num_kv_heads * sizeof(half), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_mask,  tensor_mask->data,  kv_size * PADD(batch_size, 32) * sizeof(half), cudaMemcpyHostToDevice, stream);
        // cudaMemcpyAsync(d_padded_mask,  mask_f16_padded, 32 * kv_size * sizeof(half), cudaMemcpyHostToDevice, stream);

        cudaStreamSynchronize(stream);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, stream);
        cudaMalloc((void **)&d_softmax_lse,  num_heads * batch_size * sizeof(float));
        if(flash_attn_orig) {
            flash_attn_fwd(d_query, d_key, d_value, d_mask, d_qkv, d_softmax_lse, head_dim, batch_size, kv_size, num_heads, num_kv_heads, 1, 1, scale, stream);
        } else {
            constexpr int nqpb = 16;
            constexpr int ncpw = 128;

            const int nwarps = batch_size <= nqpb ? std::max(2, std::min((int) kv_size/ncpw, 8)) : 1;

            dim3 blocks_num((batch_size + nqpb - 1) / nqpb, num_heads, 1);
            dim3 block_dim(32, nwarps, 1);

            const size_t shmem_f_ = nqpb*(head_dim + nwarps*(ncpw + nqpb))*(sizeof(float)/2);

            flash_attn_ext_f16<128, nqpb, ncpw><<<blocks_num, block_dim, shmem_f_, stream>>>(
                (const char*)d_query_f32, (const char*)d_key, (const char*)d_value, (const char*)d_mask, d_qkv_f32, scale,
                head_dim, batch_size, num_heads, 1, // query
                head_dim, kv_size, num_kv_heads, 1, // key value
                PADD(batch_size, 32), kv_size * 2, // masks
                head_dim * num_heads * 4, head_dim * 4, head_dim * batch_size * num_heads * 4, // nb query
                // head_dim * 2, head_dim * kv_size * 2, head_dim  * kv_size * num_kv_heads * 2, // nb key value
                head_dim * num_heads * 2, head_dim * 2, head_dim*kv_size * 2,
                // head_dim * 2, head_dim * kv_size * 2, head_dim * kv_size * num_heads * 2, // nb key value
                head_dim, num_heads, batch_size, 1);
        }
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);

        float millis = 0.0f;
        cudaEventElapsedTime(&millis, start, stop);

        // transfer data from device to host
        if(flash_attn_orig) {
            cudaMemcpyAsync(qkv_cuda, d_qkv, head_dim * batch_size * num_heads * sizeof(half), cudaMemcpyDeviceToHost, stream);
        } else {
            cudaMemcpyAsync(qkv_cuda_f32, d_qkv_f32, head_dim * batch_size * num_heads * sizeof(float), cudaMemcpyDeviceToHost, stream);
        }
        // cudaMemcpyAsync(key_f16, d_key, head_dim * kv_size * num_heads * sizeof(half), cudaMemcpyDeviceToHost, stream);

        cudaStreamSynchronize(stream);

        float max_diff = 0.0f;
        int head_idx = 0, batch_idx = 0, dim_idx = 0;

        if(flash_attn_orig) {
            print_array("Flash Attention", qkv_cuda, 1, 16, head_dim);
            for(int h = 0; h < num_heads; h++) {
                for(int b = 0; b < batch_size; b++) {
                    for(int i = 0; i < head_dim; i++) {
                        if(fabs(qkv[b*num_heads*head_dim + h*head_dim + i] - __half2float(qkv_cuda[b*num_heads*head_dim + h*head_dim + i])) > max_diff) {
                            max_diff = fabs(qkv[b*num_heads*head_dim + h*head_dim + i] -  __half2float(qkv_cuda[b*num_heads*head_dim + h*head_dim + i]));
                            head_idx = h;
                            batch_idx = b;
                            dim_idx = i;
                        }
                    }
                }
            }
            printf("\ncuda time: %.4f ms\n", millis);

            printf("R (%.4f) CUDA(%.4f) diff: %.4f - head = %d, batch = %d, dim = %d\n",
                qkv[batch_idx*num_heads*head_dim + head_idx*head_dim + dim_idx],  __half2float(qkv_cuda[batch_idx*num_heads*head_dim + head_idx*head_dim + dim_idx]), max_diff, head_idx, batch_idx, dim_idx);
        } else {
            print_array("Flash Attention Llama", qkv_cuda_f32, 1, 16, head_dim);
            for(int h = 0; h < num_heads; h++) {
                for(int b = 0; b < batch_size; b++) {
                    for(int i = 0; i < head_dim; i++) {
                        if(fabs(qkv[b*num_heads*head_dim + h*head_dim + i] - qkv_cuda_f32[b*num_heads*head_dim + h*head_dim + i] ) > max_diff) {
                            max_diff = fabs(qkv[b*num_heads*head_dim + h*head_dim + i] - qkv_cuda_f32[b*num_heads*head_dim + h*head_dim + i]);
                            head_idx = h;
                            batch_idx = b;
                            dim_idx = i;
                        }
                    }
                }
            }

            printf("\ncuda time: %.4f ms\n", millis);

            printf("R (%.4f) CUDA(%.4f) diff: %.4f - head = %d, batch = %d, dim = %d\n",
                qkv[batch_idx*num_heads*head_dim +  head_idx*head_dim + dim_idx],  qkv_cuda_f32[batch_idx*num_heads*head_dim +  head_idx*head_dim + dim_idx], max_diff, head_idx, batch_idx, dim_idx);
        }

        // clean up device memory
        cudaFree(d_query);
        cudaFree(d_key);
        cudaFree(d_value);
        cudaFree(d_qkv);
    }

    free(query);
    free(key);
    free(value);
    free(qkv);
    free(qkv_cuda);
}

void cuda_test() {
    // real_test();
    cap_test();
}