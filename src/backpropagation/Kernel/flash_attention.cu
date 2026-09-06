#include <iostream>
#include <iterator>
#include <math.h>
#include <mma.h>
#include <random>
#include <vector>
#include <cfloat>
#include <cstdlib>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// we will experiement with tensor cors later on :)
__global__ void matmulLastTwo4DKernel(
    float *A,      // (a, b, c, d)
    float *B,      // (a, b, d, e)
    float *C,      // (a, b, c, e)
    float scaling, // to re-use this for something like attention sore, and backpropagation.
    int a,
    int b,
    int c,
    int d,
    int e)
{
    int rows = blockIdx.y * blockDim.y + threadIdx.y;
    int cols = blockIdx.x * blockDim.x + threadIdx.x;

    if (rows >= c || cols >= e)
        return;

    int c_a = blockIdx.z / b;
    int c_b = blockIdx.z % b;

    int skipA = c_a * (b * c * d) + c_b * (c * d);
    int skipB = c_a * (b * d * e) + c_b * (d * e);

    float sum = 0.0f;
    for (int i = 0; i < d; ++i)
    {
        /*
            Little bit of a re-cap after my cooperate SWE work I might have forgotten these,
            skipA = normal offset, rows and cols are globally unqiue id, we need to skip those to reach the particular thread.

        */
        float valA = A[skipA + rows * d + i];
        float valB = B[skipB + i * e + cols];
        sum += valA * valB;
    }

    int out_idx = c_a * (b * c * e) + c_b * (c * e) + rows * e + cols;
    C[out_idx] = sum * scaling;
}

/*


The shape of matrix P is:  (batch, n_head, seq_len, seq_len) from the attn head.


I was hitting the so called "flow state" when I wrote parallel reduction for
forward pass kernels reading pdf's from NVIDA now I do not remember, but
lets try to unfold first.
*/

__device__ float warpReduceSum(float val)
{
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// One Kernel that accounts for seq_len > 32 if its small don't care
// even though we would have the advantage of warp level reduction.

// forget 1024 hardware limit for NOW at least lets get the model working atleast
// it will be a weak model but lets focus on getting the result right at first.
__global__ void softmaxBackTankKernel(
    float *P,  // (batch_size * num_heads * seq_len * seq_len )
    float *dY, // Shape (batch_size, seq_len, num_head, head_dim) again I might be wrong I am old.
    float *out,
    int N,
    int batch_size,
    int seq_len,
    int n_head)
{

    int batch_idx = blockIdx.z;
    int nhead_idx = blockIdx.y;
    int seq_len_idx1 = blockIdx.x;
    int seq_len_idx2 = threadIdx.x;

    int lane = threadIdx.x % 32;     // position within warp
    int warp_id = threadIdx.x / 32;  // position within block
    int num_warps = blockDim.x / 32; // total warps avalible

    int row_base = batch_idx * (n_head * seq_len * seq_len) + nhead_idx * (seq_len * seq_len) + seq_len_idx1 * seq_len;

    // I will note here, I am just learning, this will get populated and reduction happens in the warp level.
    __shared__ float smem_pdy[32]; // certian limit is there depending upon the GPU but this should be fine;.
    __shared__ float s_shared;

    // this is what I like to call surface level reduction
    // I have noted this concept on softmax_activation.org
    float tempSum = 0.0f;
    for (int i = seq_len_idx2; i < seq_len; i += blockDim.x)
    {
        // here i is the offset and blockDim.x is the number of thread in a block.
        tempSum += P[row_base + i] * dY[row_base + i];
    } // multipled with the upstream gradient dY, I like to call it G but, school damn.
    // I have a cheat sheet in collab somewhere.

    tempSum = warpReduceSum(tempSum);

    __syncthreads();

    if (seq_len_idx2 == 0)
    {
        // lane 0 if the each warp get assigned the reduced sum.
        smem_pdy[warp_id] = tempSum;
    } // confusing but I might get used to it, I promise, even after I wrote this from scratch twice already :)

    __syncthreads();

    float rowSum = (lane < num_warps) ? smem_pdy[lane] : 0.0f;
    if (warp_id == 0)
        rowSum = warpReduceSum(rowSum);

    if (seq_len_idx2 == 0)
        s_shared = rowSum;

    __syncthreads();

    float s = s_shared;

    if (seq_len_idx2 < seq_len)
    {
        for (int i = seq_len_idx2; i < seq_len; i += blockDim.x)
            out[row_base + i] = P[row_base + i] * (dY[row_base + i] - s);
    }
}

__device__ __forceinline__ void ParallelReducerMultiple(
    float &localSum1,
    float &localSum2)
{
    for (int offset = 16; offset > 0; offset /= 2)
    {
        localSum1 += __shfl_down_sync(0xffffffff, localSum1, offset);
        localSum2 += __shfl_down_sync(0xffffffff, localSum2, offset);
    }
    localSum1 = __shfl_sync(0xffffffff, localSum1, 0);
    localSum2 = __shfl_sync(0xffffffff, localSum2, 0);
}

__device__ __forceinline__ void ParallelReducer(float &localSum)
{
    for (int offset = 16; offset > 0; offset /= 2)
        localSum += __shfl_down_sync(0xffffffff, localSum, offset);
    localSum = __shfl_sync(0xffffffff, localSum, 0);
}

// LayerNorm Backpropagation

/*
    Eneginnering tradeoffs here, if we have that (x - u) from our forward pass kernel
    then we will need to reserve our VRAM, lets re-compute that again here.
*/

// C dimension > 32 we use shared memory here, if it was  < 32 then register could talk with each other in faster way, even if they are they wont because of this but its okay here.
// Note:- I am not so sharp and smart so I am taking my time here to derive and understand.

__global__ void LayerNormBackPropgationKernel(
    float *x,     // Shape (B, T, C)
    float *G,     // Shape (B, T, C) aftermath shape is this
    float *mc,    // mean cache (B*T, C)
    float *sdc,   // sdc cache (B*T,)
    float *gamma, // (C)
    int B,
    int T,
    int C)
{
    int batch_idx = blockIdx.y;
    int row_idx = blockIdx.x;

    int D = C;

    // skipping index formula
    float *x_row = x + batch_idx * (T * C) + row_idx * C;
    float *G_row = G + batch_idx * (T * C) + row_idx * C;

    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int num_warps = blockDim.x / 32;

    __shared__ float shared_sum_1[32];
    __shared__ float shared_sum_2[32];
    float epsilon = 1e-8f;

    float mean_val = mc[batch_idx * T + row_idx];
    float std_val = sdc[batch_idx * T + row_idx];

    // I like to call temp reduction
    float sum_term_1 = 0.0f, sum_term_2 = 0.0f;
    for (int i = threadIdx.x; i < C; i += blockDim.x)
    {
        float x_hat = (x_row[i] - mean_val) / sqrtf(std_val * std_val + epsilon);
        sum_term_1 += G_row[i] * gamma[i];
        sum_term_2 += G_row[i] * gamma[i] * x_hat;
    }

    // reduce within each wrap
    ParallelReducerMultiple(sum_term_1, sum_term_2);

    if (lane == 0)
    {
        shared_sum_1[warp_id] = sum_term_1;
        shared_sum_2[warp_id] = sum_term_2;
    }
    __syncthreads();

    if (warp_id == 0)
    {
        sum_term_1 = (lane < num_warps) ? shared_sum_1[lane] : 0.0f;
        sum_term_2 = (lane < num_warps) ? shared_sum_2[lane] : 0.0f;

        // reduce that element within that shared memory
        // when each wrap output is contributed
        ParallelReducerMultiple(sum_term_1, sum_term_2);

        if (lane == 0)
        {
            shared_sum_1[0] = sum_term_1;
            shared_sum_2[0] = sum_term_2;
        }
    }
    __syncthreads();

    // The above code should complete the one level of parallel reduction
    // we accounted for the two sum part.

    sum_term_1 = shared_sum_1[0];
    sum_term_2 = shared_sum_2[0];

    for (int i = threadIdx.x; i < C; i += blockDim.x)
    {
        // by the def sigma = sqrt(sigma + e)
        float first_component = 1.0f / (D * sqrtf(std_val * std_val + epsilon));

        // printf("First componenet: %f epsilon: %.9g sigma^2 %f sqrt(d_head) %f \n", first_component, epsilon, sdc_row[i] * sdc_row[i], D);

        float dl_x_hat = G_row[i] * gamma[i] * D;
        float curr_xhat = (x_row[i] - mean_val) / sqrtf(std_val * std_val + epsilon);

        float final_comp = first_component * (dl_x_hat - sum_term_1 - curr_xhat * sum_term_2);
        x_row[i] = final_comp;
    }
}

__global__ void LayerNormBetaGammaBackward(
    float *x,      // (B, T, C)
    float *mc,     // (B*T, C)
    float *sdc,    // (B*T, C)
    float *G,      // Shape(B, T, C)
    float *dbeta,  // [c]
    float *dgamma, // [c]
    int B,
    int T,
    int C)
{
    int batch_idx = blockIdx.y;
    int row_idx = blockIdx.x;

    float *x_row = x + batch_idx * (T * C) + row_idx * C;
    float *G_row = G + batch_idx * (T * C) + row_idx * C;

    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int num_warps = blockDim.x / 32;

    float epsilon = 1e-8f;

    float mean_val = mc[batch_idx * T + row_idx]; // shapes (B*T, C)
    float std_val = sdc[batch_idx * T + row_idx];

    float x_hat = 0.0f;

    // one block per channel
    for (int i = threadIdx.x; i < C; i += blockDim.x)
    {
        float x_hat = (x_row[i] - mean_val) / sqrtf(std_val * std_val + epsilon);

        // dgamma[i] = dgamma[i] + value;
        atomicAdd(&dgamma[i], G_row[i] * x_hat);
        atomicAdd(&dbeta[i], G_row[i]);
    }
}

__global__ void sumBTC3TensorKernel(
    float *A, // Shape (B, T, C)
    float *B,
    float *C,
    float *Out,
    int batch_size,
    int seq_len,
    int d_model,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        Out[idx] = A[idx] + B[idx] + C[idx];
    }
}

__global__ void ReformBNTH_BTC_Kernel(
    float *arr, // [batch_size, num_head, seq_len, head_dim]
    float *out, // (B, T, C)
    int batch_size,
    int seq_len,
    int d_model,
    int num_head,
    int head_dim)
{
    int batch_idx = blockIdx.z;
    int num_head_idx = blockIdx.y;
    int seq_idx = blockIdx.x;

    // blockDim.x is just 256 in our case
    // here each thread will touch one hd_idx if smaller tensor
    for (int hd_idx = threadIdx.x; hd_idx < head_dim; hd_idx += blockDim.x)
    {
        int idx = batch_idx * (num_head * seq_len * head_dim) +
                  num_head_idx * (seq_len * head_dim) +
                  seq_idx * (head_dim) + hd_idx;
        // d_model = num_head * seq_len
        int c_idx = num_head_idx * head_dim + hd_idx;
        int outIdx = batch_idx * (seq_len * d_model) + seq_idx * (d_model) + c_idx;

        out[outIdx] = arr[idx];
    }
}

extern "C"
{

    void layer_norm_beta_gamma_backward(
        float *x,      // (B, T, C)
        float *mc,     // (B*T, C)
        float *sdc,    // (B*T, C)
        float *G,      // Shape(B, T, C)
        float *dbeta,  // [c]
        float *dgamma, // [c]
        int B,
        int T,
        int C)
    {
        dim3 blockDim(256, 1, 1);
        // TxB block each one havong 256 threads assigned to them
        dim3 gridDim(T, B, 1);

        LayerNormBetaGammaBackward<<<gridDim, blockDim>>>(x, G, mc, sdc, dgamma, dbeta, B, T, C);

        cudaDeviceSynchronize();
    }
    void layernorm_backward(
        float *x,
        float *G,
        float *mc,
        float *sdc,
        float *gamma,
        float D,
        int B,
        int T,
        int C)
    {
        dim3 blockDim(256, 1, 1);
        dim3 gridDim(T, B, 1); // one block per (batch, row)

        LayerNormBackPropgationKernel<<<gridDim, blockDim>>>(
            x, G, mc, sdc, gamma, B, T, C);

        cudaDeviceSynchronize();
    }

    void addThreeTensor(
        float *A,
        float *B,
        float *C,
        float *Out,
        int batch_size,
        int seq_len,
        int d_model)
    {
        int threads = 256;
        int N = batch_size * seq_len * d_model;
        int blocks = (N + threads - 1) / threads;

        sumBTC3TensorKernel<<<blocks, threads>>>(A, B, C, Out, batch_size, seq_len, d_model, N);

        cudaDeviceSynchronize();
    }

    void ReformBNTH_BTC(
        float *arr, // [batch_size, num_head, seq_len, d_head]
        float *out, // (B, T, C)
        int batch_size,
        int seq_len,
        int d_model,
        int num_head,
        int head_dim)
    {
        int threads_per_block = 256;
        dim3 grid(seq_len, num_head, batch_size);
        dim3 block(threads_per_block);

        ReformBNTH_BTC_Kernel<<<grid, block>>>(arr, out, batch_size, seq_len, d_model, num_head, head_dim);

        cudaDeviceSynchronize();
    }
    void layerNormBackGrad(
        float *x,     // (B, T, C) input from forward pass
        float *G,     // (B, T, C) upstream gradient dL/dy
        float *mc,    // mean cache (B*T,)    ONE float per row, not per channel
        float *sdc,   // std dev cache (B*T,) ONE float per row, not per channel
        float *gamma, // (C,)  learnable scale
        int B,
        int T,
        int C)
    {
        int row_count = B * T;                            // one row per (batch, token) pair
        int threads_per_block = ((C + 31) / 32) * 32;     // round C up to nearest warp (32)
        threads_per_block = min(threads_per_block, 1024); // cap at CUDA's max threads/block

        dim3 blockSize(threads_per_block, 1, 1);
        dim3 gridSize(row_count, 1, 1); // one block per row, simple 1D grid

        // according to my common sense and schooling D is the total number of element

        LayerNormBackPropgationKernel<<<gridSize, blockSize>>>(
            x, G, mc, sdc, gamma, B, T, C);

        cudaDeviceSynchronize();
    }
    void softmaxBackGradKernel(
        float *P,
        float *dY,
        float *out,
        int N,
        int batch_size,
        int seq_len,
        int n_head)
    {
        dim3 blockSize(((seq_len + 31) / 32) * 32, 1, 1); // enough threads to cover one row, e.g. round seq_len up to nearest 32
        dim3 gridSize(
            seq_len,
            n_head,
            batch_size);

        softmaxBackTankKernel<<<gridSize, blockSize>>>(
            P, dY, out, N, batch_size, seq_len, n_head);
        cudaDeviceSynchronize();
    }

    void MatMul4D(
        float *A,      // (a, b, c, d)
        float *B,      // (a, b, d, e)
        float *C,      // (a, b, c, e)
        float scaling, // pass 1.0f if not scaling
        int a,
        int b,
        int c,
        int d,
        int e)

    {
        dim3 blockDim(16, 16, 1);
        dim3 gridDim(
            (e + blockDim.x - 1) / blockDim.x, // cols = e
            (c + blockDim.y - 1) / blockDim.y, // rows = c
            a * b                              // combined batch*head axis
        );

        matmulLastTwo4DKernel<<<gridDim, blockDim>>>(
            A, B, C, scaling,
            a, b, c, d, e);
        cudaDeviceSynchronize();
    }
}