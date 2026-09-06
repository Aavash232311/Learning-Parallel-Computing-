#pragma once
#include "../include/helper.hpp"
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <cstdio>
#include <chrono>

#include "./interface_back.hpp"
#include "./flash_attention.hpp"
#include "./flash_attention_linear.hpp"

#include "../include/utils.hpp"
#include "../include/linear.hpp"
#include "../include/p_head.hpp"
#include "../include/cache_in.hpp"
#include "../include/cache_out.hpp"
#include "../include/netattention.hpp"
#include "../include/attention_params.hpp"
#include "../include/single_embeddings.hpp"

using namespace std;

// Releases debugger script for python project to check
// Autograd class and its child classes are getting messy

class AutogradEngineDebuggerRelease : public FlashAttention, public FlashAttentionLinear
{
public:
    AutogradEngineDebuggerRelease(int d_model, int vocab_size, int num_heads,
                                  int seq_len, int batch_size, bool debug)
        : AutoGradEngine(d_model, vocab_size, num_heads, seq_len, batch_size, debug),
          FlashAttention(d_model, vocab_size, num_heads, seq_len, batch_size, debug),
          FlashAttentionLinear(d_model, vocab_size, num_heads, seq_len, batch_size, debug)
    {
    }

    /**
     * @class pyDebuggerReleaseStage1
     * @brief Releases paramaters y, y predicted, delta, CE+softmax backpropagation, h (from lm head)
     *
     * Simply for stage one releases the above paramaters from the attention interface class
     *
     * @note Call this only after dl_dz_upstream_gradient() and before gradient_linear <- tranposes h^T
     * @warning Do not call this before the above methods are called as they tend to write garbage data.
     */
    void pyDebuggerReleaseStage1()
    {
        // release those paramaters so that we can cross verify the kernel in python
        // now I wnat to debug along rather than estimating by judging few rows and cols.
        // pointers are manupluated and copied to the host.

        // y_actual and y_predicted are in GDDR VRAM
        // for debugging its fine to allocate here and free it

        float *host_y_actual = (float *)malloc(batch_size * seq_len * vocab_size * sizeof(float));
        float *host_y_prediced = (float *)malloc(batch_size * seq_len * vocab_size * sizeof(float));

        // happens only one that that also for debugging so it should be fine.
        cudaMemcpy(host_y_actual, model_paramaters.y_actual, batch_size * seq_len * vocab_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_y_prediced, model_paramaters.y_predicted, batch_size * seq_len * vocab_size * sizeof(float), cudaMemcpyDeviceToHost);

        bulkRelease<float>({{host_y_actual, batch_size * seq_len * vocab_size, "y_actual.bin"},
                            {host_y_prediced, batch_size * seq_len * vocab_size, "y_prediced.bin"},
                            {model_paramaters.dl_dz_out_host, batch_size * seq_len * vocab_size, "delta.bin"},
                            {model_paramaters.h, batch_size * seq_len * d_model, "h.bin"}});

        // if (debug)
        // {
        //     cout << "delta" << endl;
        //     utils->printFlatArray3D(model_paramaters.dl_dz_out_host, batch_size,
        //     seq_len, vocab_size);
        // }

        free(host_y_actual);
        free(host_y_prediced);
    }

    /**
     * @class pyDebuggerReleaseStage2
     * @brief Releases paramaters h^T and dl/dw = del h^T
     *
     * Simply for stage one releases the above paramaters from the attention interface class
     *
     * @note Call this only after gradient_linear and dl_dw_upstream_gradient
     * @warning Do not call this before the above methods are called as they tend to write garbage data.
     */
    void pyDebuggerReleaseStage2()
    {
        // After the gradient_linear() gets called model_paramaters.h gets written
        bulkRelease<float>({{model_paramaters.h, batch_size * seq_len * d_model, "h_t.bin"},
                            {model_paramaters.dl_dw_host, batch_size * d_model * vocab_size, "dl_dw.bin"}}); // out delta h^T binary
                                                                                                             // second stage release for the autograd engine.
    }

    /**
     * @class pyDebuggerReleaseStage3
     * @brief Releases paramaters w^t, w and dl_dh (Which is the upstream gradient G)
     *
     * Simply for stage one releases the above paramaters from the attention interface class
     *
     * @note Can call this in any order because w is independent of the result from pervious things like h.
     */

    void pyDebuggerReleaseStage3()
    {
        // copy w^T to host for the python script to read the binary

        float *wt_host = (float *)malloc(d_model * vocab_size * sizeof(float));
        float *dl_dh_host = (float *)malloc(batch_size * seq_len * d_model * sizeof(float));

        cudaMemcpy(wt_host, model_paramaters.wt_out_d, d_model * vocab_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(dl_dh_host, model_paramaters.Contact_G_Upstream, batch_size * seq_len * d_model * sizeof(float), cudaMemcpyDeviceToHost);

        bulkRelease<float>(
            {
                {wt_host, d_model * vocab_size, "wt.bin"},
                {model_paramaters.w_host, d_model * vocab_size, "w.bin"},
                // for now this is the G shape (B, T, C)
                {dl_dh_host, batch_size * seq_len * d_model, "dl_dh.bin"},
            });

        // if (debug)
        // {
        //     cout << "W^T from C++" << endl;
        //     utils->printFlatArray2D(wt_host, vocab_size, d_model);
        // }
        free(dl_dh_host);
        free(wt_host);
    }

    /**
     * @class pyDebuggerReleaseStage4
     * @brief Releases paramaters P, V for now, reason this is a seperate method is because we might have something else to release in future.
     *
     *
     * Simply for stage one releases the above paramaters from the attention interface class
     *
     * @note Can call this in any order because w is independent of the result from pervious things like h.
     */

    void pyDebuggerReleaseStage4()
    {

        bulkRelease<float>(
            {
                {model_paramaters.attention_head.P, batch_size * num_heads * seq_len * seq_len, "P.bin"},
                {model_paramaters.attention_head.V, batch_size * num_heads * seq_len * head_dim, "V.bin"},
            });
    }

    /**
     * @class pyDebuggerReleaseStage5
     * @brief Releases paramaters P^T and V^T for back most layer of the flash attention, also un-contact G of 3D tensor
     * Also UNCONTACTS UPSTREAM GRADIENT G, from (B, T, C) to (batch_size, seq_len, num_head, head_dim)
     *
     * These methods are in sequential order, so this releases the P^T and V^T for a python debugger to verify and check
     * model_paramaters.attention_head.P and model_paramaters.attention_head.V should be consumed by the transpose Kernel.
     * @note Call this after all the 3, 2, 1 stage are released
     */
    void pyDebuggerReleaseStage5()
    {
        // Uncontact_G_Upstream is in Global mem
        float *UncG_host = (float *)malloc(batch_size * seq_len * num_heads * head_dim * sizeof(float));
        float *dP = (float *)malloc(batch_size * num_heads * seq_len * seq_len * sizeof(float));
        float *dV = (float *)malloc(batch_size * num_heads * seq_len * head_dim * sizeof(float));

        float *Pt = (float *)malloc(batch_size * num_heads * seq_len * seq_len * sizeof(float));
        float *Vt = (float *)malloc(batch_size * num_heads * head_dim * seq_len * sizeof(float));

        cudaMemcpy(UncG_host, model_paramaters.Uncontact_G_Upstream, batch_size * seq_len * num_heads * head_dim * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(dP, model_paramaters.dP, batch_size * num_heads * seq_len * seq_len * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(dV, model_paramaters.dV, batch_size * num_heads * seq_len * head_dim * sizeof(float), cudaMemcpyDeviceToHost);

        cudaMemcpy(Pt, model_paramaters.P_T_device_out, batch_size * num_heads * seq_len * seq_len * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(Vt, model_paramaters.V_T_device_out, batch_size * num_heads * head_dim * seq_len * sizeof(float), cudaMemcpyDeviceToHost);

        bulkRelease<float>(
            {{Pt, batch_size * num_heads * seq_len * seq_len, "pt.bin"},
             {Vt, batch_size * num_heads * head_dim * seq_len, "vt.bin"}, // keep in mind of the transposed shape here
             {UncG_host, batch_size * seq_len * num_heads * head_dim, "G_uncontact.bin"},
             {dP, batch_size * num_heads * seq_len * seq_len, "dp.bin"},
             {dV, batch_size * num_heads * seq_len * head_dim, "dv.bin"}});

        // after part I release we are going to release
        free(UncG_host);
        free(dP);
        free(dV);

        free(Pt);
        free(Vt);
    }
    /**
     * @class pyDebuggerReleaseStage6
     * @brief Releases paramaters upstream gradient from the softmax kernel


     * @note Call this after all the 1, 2, 3, 4, and 5 are called stage are released
     */

    void pyDebuggerReleaseStage6()
    {
        float *G = (float *)malloc(batch_size * num_heads * seq_len * seq_len * sizeof(float));
        cudaMemcpy(G, d_scores_device, batch_size * num_heads * seq_len * seq_len * sizeof(float), cudaMemcpyDeviceToHost);

        // Note:- in P = softmax(S)
        // Normal to see bunch of zeros because of softmax
        // make sure the row sums to one

        // this gets called only when debugger flag is on remember?
        // std::cout << "From the kernel itself " << std::endl;
        // utils->print2DMatrixLastTwo(

        //     G,
        //     batch_size,
        //     num_heads,
        //     seq_len,
        //     seq_len);

        bulkRelease<float>(
            {{G, batch_size * num_heads * seq_len * seq_len * sizeof(float), "softmax_upstream.bin"}});

        free(G);
    }

    /**
     * @class pyDebuggerReleaseStage7
     * @brief Releases (1/sqrt(dk) GK) (1/sqrt(dk)) G^T Q d_score_t for the dK = N G^T Q part.
     * std dev and mean from the forward pass


    * @note Call this after all the 1, 2, 3, 4, 5, 6 and 7 are called stage are released
    */

    void pyDebuggerReleaseStage7()
    {
        float *dQ_host = (float *)malloc(batch_size * seq_len * num_heads * head_dim * sizeof(float));
        float *dK_host = (float *)malloc(batch_size * seq_len * num_heads * head_dim * sizeof(float));

        float *Q_host = (float *)malloc(batch_size * seq_len * num_heads * head_dim * sizeof(float));
        float *K_host = (float *)malloc(batch_size * seq_len * num_heads * head_dim * sizeof(float));
        float *d_score_t = (float *)malloc(batch_size * num_heads * seq_len * seq_len * sizeof(float));

        float *mean_host = (float *)malloc(batch_size * seq_len * d_model * sizeof(float));
        float *std_dev_host = (float *)malloc(batch_size * seq_len * d_model * sizeof(float));
        float *gamma_host = (float *)malloc(d_model * sizeof(float));

        cudaMemcpy(dQ_host, model_paramaters.dQ, batch_size * seq_len * num_heads * head_dim * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(dK_host, model_paramaters.dK, batch_size * seq_len * num_heads * head_dim * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(Q_host, model_paramaters.attention_head.Q_cache, batch_size * seq_len * num_heads * head_dim * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(K_host, model_paramaters.attention_head.K_cache, batch_size * seq_len * num_heads * head_dim * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(d_score_t, model_paramaters.d_score_t, batch_size * num_heads * seq_len * seq_len * sizeof(float), cudaMemcpyDeviceToHost);

        cudaMemcpy(mean_host, model_paramaters.attention_head.mean_cache, batch_size * seq_len * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(std_dev_host, model_paramaters.attention_head.std_dev_cache, batch_size * seq_len * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(gamma_host, model_paramaters.attention_head.d_gamma, d_model * sizeof(float), cudaMemcpyDeviceToHost);

        bulkRelease<float>(
            {{dQ_host, batch_size * seq_len * num_heads * head_dim, "dq.bin"},
             {dK_host, batch_size * seq_len * num_heads * head_dim, "dk.bin"},
             {Q_host, batch_size * seq_len * num_heads * head_dim, "q.bin"},
             {K_host, batch_size * seq_len * num_heads * head_dim, "k.bin"},
             {mean_host, batch_size * seq_len, "mean_cache.bin"},
             {std_dev_host, batch_size * seq_len, "std_dev_cache.bin"},
             {gamma_host, d_model * sizeof(float), "gamma_host.bin"},
             {d_score_t, batch_size * num_heads * seq_len * seq_len, "d_score_t.bin"}});

        free(d_score_t);
        free(dQ_host);
        free(Q_host);
        free(K_host);
        free(dK_host);

        free(mean_host);
        free(std_dev_host);
        free(gamma_host);
    }

    /**
     * @class pyDebuggerReleaseStage8
     * @brief Releases Weights of QKT transposed, shape of upstream dQ, dK, dV into BTC, G_x_hat


    * @note Releases Weights of QKT transposed
    */

    void pyDebuggerReleaseStage8()
    {
        float *WQT = (float *)malloc(d_model * d_model * sizeof(float));
        float *WKT = (float *)malloc(d_model * d_model * sizeof(float));
        float *WVT = (float *)malloc(d_model * d_model * sizeof(float));

        float *upQ;
        float *upK;
        float *upV;

        float *G_x_hat_host;

        float *layer_norm_x;

        float *d_beta;  

        upQ = (float *)malloc(batch_size * seq_len * d_model * sizeof(float));
        upK = (float *)malloc(batch_size * seq_len * d_model * sizeof(float));
        upV = (float *)malloc(batch_size * seq_len * d_model * sizeof(float));

        d_beta = (float *)malloc(batch_size * sizeof(float));

        G_x_hat_host = (float *)malloc(batch_size * seq_len * d_model * sizeof(float));

        layer_norm_x = (float *)malloc(batch_size * seq_len * d_model * sizeof(float));;

        cudaMemcpy(WQT, model_paramaters.WqT, d_model * d_model * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(WKT, model_paramaters.WkT, d_model * d_model * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(WVT, model_paramaters.WvT, d_model * d_model * sizeof(float), cudaMemcpyDeviceToHost);

        cudaMemcpy(upQ, model_paramaters.qUp, batch_size * seq_len * d_model * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(upK, model_paramaters.kUp, batch_size * seq_len * d_model * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(upV, model_paramaters.vUp, batch_size * seq_len * d_model * sizeof(float), cudaMemcpyDeviceToHost);

        cudaMemcpy(G_x_hat_host, model_paramaters.G_x_hat, batch_size * seq_len * d_model * sizeof(float), cudaMemcpyDeviceToHost);

        cudaMemcpy(layer_norm_x, model_paramaters.attention_head.x, batch_size * seq_len * d_model * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(d_beta, model_paramaters.attention_head.d_beta, d_model * sizeof(float), cudaMemcpyDeviceToHost);

        // Also release the weight should be on host from Linear class, Later we will think of a way to
        // reduce memory copy on PCIe BUS which is costly under each epoch.

        bulkRelease<float>(
            {
                {WQT, d_model * d_model, "wqt.bin"},
                {WKT, d_model * d_model, "wkt.bin"},
                {WVT, d_model * d_model, "wvt.bin"},
                {model_paramaters.attention_head.host_WK, d_model * d_model, "wq.bin"},
                {model_paramaters.attention_head.host_WQ, d_model * d_model, "wk.bin"},
                {model_paramaters.attention_head.host_WV, d_model * d_model, "wv.bin"},
                {upQ, batch_size * seq_len * d_model, "upQ.bin"},
                {upK, batch_size * seq_len * d_model, "upK.bin"},
                {upV, batch_size * seq_len * d_model, "upV.bin"},
                {G_x_hat_host, batch_size * seq_len * d_model, "G_x_hat.bin"},
                {layer_norm_x, batch_size * seq_len * d_model, "layer_norm_x.bin"},
                {d_beta, d_model, "beta.bin"}
            });

        free(WQT);
        free(WKT);
        free(WVT);

        free(upQ);
        free(upK);
        free(upV);

        free(G_x_hat_host);
        free(layer_norm_x);
        free(d_beta);
    }
};