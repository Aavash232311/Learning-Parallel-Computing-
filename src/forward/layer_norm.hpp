#pragma once
#include "../include/utils.hpp"
#include "../include/helper.hpp"
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <cstdio>
#include <chrono>


using namespace std;

extern "C" void layerNormalization(float *x, float *gamma, float *beta, float *std_dev_cache, float *mean_cache, int batch_size, int seq_len, int d_model);

class LayerNorm
{
    // we can loose the symmantic meaning in the norm procress so gamma and beta as learnable parms adjusts accordingly.
    float *h_gamma = nullptr;
    float *h_beta = nullptr;

    float *d_gamma;
    float *d_beta;

    float *d_x;

    int batch_size;
    int seq_len;
    int d_model;

    float *std_dev_cache;
    float *mean_cache;

    bool debug = true;

    unique_ptr<Utility> utils;

public:
    LayerNorm(int batch_size, int seq_len, int d_model) // LayerNorm(x) = γ . (x - μ) / √(σ² + ε) + β
    {
        h_gamma = (float *)malloc(d_model * sizeof(float));
        h_beta = (float *)malloc(d_model * sizeof(float));

        this->batch_size = batch_size;
        this->seq_len = seq_len;
        this->d_model = d_model;

        utils = make_unique<Utility>();

        for (int i = 0; i < d_model; ++i)
        {
            h_gamma[i] = 1.0f; // because these are the learnable paramaters
            h_beta[i] = 0.0f;
        }

        // allocate memory for the device
        cudaMalloc((void **)&d_gamma, d_model * sizeof(float));
        cudaMalloc((void **)&d_beta, d_model * sizeof(float));
        cudaMalloc((void **)&d_x, batch_size * seq_len * d_model * sizeof(float));

        // copy to device for both now, should be done in constuctor to get the performace advantage.
        // H2D copy
        cudaMemcpy(d_gamma, h_gamma, d_model * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_beta, h_beta, d_model * sizeof(float), cudaMemcpyHostToDevice);

        cudaMalloc((void **)&std_dev_cache, batch_size * seq_len * sizeof(float));

        cudaMalloc((void **)&mean_cache, batch_size * seq_len * sizeof(float));
    }

    ~LayerNorm()
    {
        free(h_gamma);
        free(h_beta);

        cudaFree(d_beta);
        cudaFree(d_gamma);
        cudaFree(d_x);

        cudaFree(std_dev_cache);
        cudaFree(mean_cache);
    }

    void forward(float *x) // after adding the embeddings we have (B, T, C) shape
    {
        // we first normalize and then pass it to the transofmer which is normal in modern transformer.
        // we also have to be very precise in cases like this because the bugs are scilent and we would never know what happened.

        // first we need to calculate the mean μ
        // second standard deviation
        // then normalize gama and beta are the learnable paramaters.

        // copy x to device
        cudaMemcpy(d_x, x, batch_size * seq_len * d_model * sizeof(float), cudaMemcpyHostToDevice);
        layerNormalization(d_x, d_gamma, d_beta, std_dev_cache, mean_cache, batch_size, seq_len, d_model);
        cudaMemcpy(x, d_x, batch_size * seq_len * d_model * sizeof(float), cudaMemcpyDeviceToHost);

        // if (debug)
        // {
        //     float *mean_host = (float *)malloc(batch_size * seq_len * d_model * sizeof(float));
        //     float *std_dev_host = (float *)malloc(batch_size * seq_len * d_model * sizeof(float));

        //     cudaMemcpy(mean_host, mean_cache, batch_size * seq_len * d_model * sizeof(float), cudaMemcpyDeviceToHost);
        //     cudaMemcpy(std_dev_host, std_dev_cache, batch_size * seq_len * d_model * sizeof(float), cudaMemcpyDeviceToHost);
            
        //     utils->printFlatArray2D(mean_host, batch_size * seq_len, d_model);
        //     utils->printFlatArray2D(std_dev_host, batch_size * seq_len, d_model);

        //     free(mean_host);
        //     free(std_dev_cache);
        // }

        debug = false;
    }

    float *getGammaHost()
    {
        return this->h_gamma; // of course these are tuneable.
    }

    float *getGammaDevice() // for the most part we only care about whats insdie of the GPU
    {
        return this->d_gamma;
    }

    float *getGamma()
    {
        return this->h_gamma;
    }

    float *getBetta()
    {
        return this->h_beta;
    }

    float *getStdDev()
    {
        return this->std_dev_cache;
    }

    float *getMean()
    {
        return this->mean_cache;
    }

    float *getBeta()
    {
        return this->d_beta;
    }
};