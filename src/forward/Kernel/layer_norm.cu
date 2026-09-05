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

__device__ __forceinline__ void ParallelReducer(float &localSum)
{
    for (int offset = 16; offset > 0; offset /= 2)
        localSum += __shfl_down_sync(0xffffffff, localSum, offset);
    localSum = __shfl_sync(0xffffffff, localSum, 0);
}

// We need to optimize this tomorrow it wont work if the d_model > 32
// True level of optimization without very much to loose takes more time probally thousnads of line and insanely complicated code.
__global__ void layerNormKernelSlow( // not "slow" but slower than which utilizes registers
    float *x,                        // (B, T, C)
    float *gamma,
    float *beta,
    float *std_dev_cache,
    float *mean_cache,
    int batch_size,
    int seq_len,
    int d_model)
{
    int batch_idx = blockIdx.y;
    int row_idx = blockIdx.x;

    const float *row = x + batch_idx * (seq_len * d_model) + row_idx * d_model;
    // out_row == row, x is overwritten
    float *out_row = x + batch_idx * (seq_len * d_model) + row_idx * d_model;

    int lane = threadIdx.x % 32;     // position within warp
    int warp_id = threadIdx.x / 32;  // position within block
    int num_warps = blockDim.x / 32; // total warps avalible

    __shared__ float shared_sum[32];

    float sum = 0.0f;

    for (int i = threadIdx.x; i < d_model; i += blockDim.x)
    {
        sum += row[i]; // so sum of elements that it tocuehd.
    }

    // till here the sum is a partial sum from each of the threads
    ParallelReducer(sum); // we have finint amount of warp at max size of 32, and you need to maybe add that to warp and then reduce the warp as well

    if (lane == 0)
    {
        shared_sum[warp_id] = sum; // you have that in warp level
    }

    __syncthreads();

    if (warp_id == 0)
    {
        sum = (lane < num_warps) ? shared_sum[lane] : 0.0f;

        ParallelReducer(sum); // for the sum in thread

        if (lane == 0)
        {
            shared_sum[0] = sum;
        }
    }

    __syncthreads();
    // shared_sum[0] = full sum here
    float mean = shared_sum[0] / d_model;

    // we need to pass again for the variance dont know if this is the correct way but should work

    float var_sum = 0.0f;

    for (int i = threadIdx.x; i < d_model; i += blockDim.x)
    {
        float diff = row[i] - mean;
        var_sum += diff * diff;
    } // this I like to call reduction in surface

    ParallelReducer(var_sum);

    if (lane == 0)
    {
        shared_sum[warp_id] = var_sum;
    }

    __syncthreads();

    if (warp_id == 0)
    {
        var_sum = (lane < num_warps) ? shared_sum[lane] : 0.0f;
        // we have written in the warp for that variance
        // becuase the literal formula is current - mean

        ParallelReducer(var_sum); // recude from the shared memory

        if (lane == 0)
        {
            shared_sum[0] = var_sum;
        }
    }

    __syncthreads();

    // At this point our reduction for sum works

    float variance = shared_sum[0] / d_model; // mean of that var * var
    float std = sqrtf(variance + 1e-8f);

    // y = gamma * (x - mean) / sqrt(variance + epsilon) + beta

    for (int i = threadIdx.x; i < d_model; i += blockDim.x)
    {
        out_row[i] = gamma[i] * ((row[i] - mean) / std) + beta[i];
    }

    // Shape of these two is (B*T, C)
    std_dev_cache[batch_idx * seq_len + row_idx] = std;
    mean_cache[batch_idx * seq_len + row_idx] = mean;
}

// ------------ We forgoet to account for d_model > 32 ------------- we need to do it all the time whenever using memory from the register.
__global__ void layerNormKernel(
    float *x, // (batch_size, seq_len, d_model)
    float *gamma,
    float *beta,
    float *std_dev_cache,
    float *mean_cache,
    int batch_size,
    int seq_len,
    int d_model)
{
    int seq_idx = blockIdx.x;   // which sequence position
    int batch_idx = blockIdx.y; // which batch item
    int e = threadIdx.x;        // which embed dimension, dmodel

    int idx = (seq_idx * batch_size + batch_idx) * d_model + e;

    float val = x[idx];

    float sum = val;

    // the memory can be shared across 32 thread called warp
    for (int offset = 16; offset > 0; offset /= 2)
    {
        // current idx value is incrementing from values from 4 lanes away
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // lets calculate the mean, if something is not okay here transformer will never work
    // and I will get lost in thoushands of lines of code so each module/cuda kernel should be checked very precisly.

    float mean = __shfl_sync(0xffffffff, sum, 0) / d_model; // basically after the parallel reduction the values are shifed along

    float difference = val - mean; // (x - u)

    float variance = difference * difference;

    for (int offset = 16; offset > 0; offset /= 2)
    {
        variance += __shfl_down_sync(0xffffffff, variance, offset);
    }
    variance = __shfl_sync(0xffffffff, variance, 0) / d_model;

    // normalize, that small constant Eo is used to added to prevent division from zero
    // like in the Columb's law.
    float std = sqrtf(variance + 1e-8f);
    x[idx] = gamma[e] * ((val - mean) / std) + beta[e]; // fingers crossed no race condition.

    std_dev_cache[idx] = std;
    mean_cache[idx] = mean;
}

// I must have forgotten something here,
// its been long since summer internship so we have warp level reduction here.
// if the thread is < 32, again I wont remeber in an instance just by looking at it.

extern "C"
{
    void layerNormalization(
        float *x, // (batch_size, seq_len, d_model)
        float *gamma,
        float *beta,
        float *std_dev_cache,
        float *mean_cache,
        int batch_size,
        int seq_len,
        int d_model)
    {

        if (d_model > 32)
        {
            // (batch_size, seq_len, d_model)
            dim3 grid(seq_len, batch_size);
            dim3 block(min(1024, d_model));

            layerNormKernelSlow<<<grid, block>>>(x, gamma, beta, std_dev_cache, mean_cache, batch_size, seq_len, d_model);
        }
        else
        {
            dim3 grid(seq_len, batch_size);
            dim3 block(d_model);

            layerNormKernel<<<grid, block>>>(x, gamma, beta, std_dev_cache, mean_cache, batch_size, seq_len, d_model);
        }

        cudaDeviceSynchronize();
    }
}