#pragma once
#include "../include/helper.hpp"
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <cstdio>
#include <chrono>

#include "../include/utils.hpp"
#include "../forward/linear.hpp"
#include "../include/linear.hpp"
#include "../include/p_head.hpp"
#include "../include/cache_in.hpp"
#include "../include/cache_out.hpp"
#include "../forward/layer_norm.hpp"
#include "../forward/embeddings.hpp"
#include "../include/data_loader.hpp"
#include "../include/netattention.hpp"
#include "../forward/attention_head.hpp"
#include "../include/single_embeddings.hpp"
#include "../backpropagation/interface_back.hpp"
#include "../backpropagation/flash_attention.hpp"
#include "../backpropagation/debugger_release.hpp"

extern "C" void softmax2D(float *arr, float *out, int batch_size, int seq_len, int vocab_size);
extern "C" void CrossEntropy(float *x, int *y, float *oneHotOut, float *lossOut, int batch_size, int seq_len, int vocab_size);

class AttentionInterface
{

    int d_model;
    int num_heads;
    int vocab_size;
    int batch_size;
    int seq_len;
    int head_dim;
    bool runDebugger = false;
    bool drop_last = true;

    const std::vector<int> &data;
    std::unique_ptr<Attention> attention;
    std::unique_ptr<DataLoader> dataLoader;
    std::unique_ptr<Utility> utils;

    // turn those result into proballity score
    std::unique_ptr<Linear> lm_head;

    bool debug = true;

    // DEVICE SFOTMAX BEFORE CROSS ENTROPY LOSS IN
    float *DeviceSoftmaxBLin;
    // DEVICE SOFTMAX BEFORE CORSS ENTROPY LOSS OUT
    float *DeviceSoftmaxBLout;

    // ALLOCATE MEMORY FOR Y
    float *yHotEncodeDeviceOut;
    float *outCrossEntropyDevice;

    float *outCrossEntropyHost;

    int *deviceY;

    // ------ For testing one hot encode kernel --------- //
    float *outHotEncodeOut = nullptr;

    std::unique_ptr<AutogradEngineDebuggerRelease> autograd;

    // LOW LEVEL BY DESIGN IS LITTLE BIT MESSY COMES WITH A TRADE OFFS ANYWAY

    float *dl_dz_out_device;
    float *dl_dz_out_host;

    float *dl_dw_out_device;
    float *dl_dw_out_host;

    // GPU buffer for the autograd engine in the attention interface.
    FlashAttentionPointers modelParamaters;

    // ---------- Autograd engine declaration --------------------
    float *out_h;

    // even for SWE after years of building things on my own
    // now I realize how stupid I was, sure this is the dumbest possible thing
    // goal is not perfection here, I just want to make it work

    // ---------- Autograd engine weight lm head ---------------------
    float *w_device;
    float *w_out_d;

    // For the net gradient we also want to allocate for
    // WkT and WvT

    float *WkT;
    float *WvT;
    float *WqT;

    float *Wk;
    float *WV;
    float *WQ;

    // ------------ Allocation fhas attention class ---------------------

    float *S_device;
    float *P_device;
    float *O_device;

    // P^T and V^T device alloication
    float *P_T_device;
    float *V_T_device;

    float *P_T_device_out;
    float *V_T_device_out;

    float *Uncontact_G_Upstream;
    float *Contact_G_Upstream;

    float *dV; // O = PV one of the derivative terms.
    float *dP; // same derviative terms.

    float *dQ;
    float *dK;

    // softmax activation derivative terms.

    float *ppt;

    float *d_score_t;

    /*
        Net G for layer norm very end formula in
        flashback.md
    */

    float *qUp;
    float *kUp;
    float *vUp;

    float *dqWt;
    float *dkWt;
    float *dvWt;

    float *G_x_hat; // total added up tensor

    float *device_x;

    float *dbeta;
    float *dgamma;

private:
    // ----------- TEMPORARY DEBUGGER SCRIPT ---------------------

    // B,T,C shape use if you want to see and inspeace device
    void DebugBTCFlatArray3D(
        float *d_arr,
        int B,
        int T,
        int C // vocab size
    )
    {
        float *h_arr = (float *)malloc(B * T * C * sizeof(float));

        cudaMemcpy(h_arr, d_arr, B * T * C * sizeof(float), cudaMemcpyDeviceToHost);

        utils->printLastOneOf3D(h_arr, B, T, C);

        free(h_arr);
    }

public:
    AttentionInterface(
        int d_model,
        int num_heads,
        int batch_size,
        int seq_len,
        bool drop_last,
        const std::vector<int> &data, bool debug) : data(data)
    {
        this->d_model = d_model;
        this->vocab_size = data.size();
        this->batch_size = batch_size;
        this->seq_len = seq_len;
        this->drop_last = drop_last;
        this->num_heads = num_heads;
        this->debug = debug;

        this->head_dim = d_model / num_heads;

        utils = std::make_unique<Utility>();

        attention = std::make_unique<Attention>(
            d_model,
            vocab_size,
            num_heads,
            seq_len,
            batch_size,
            debug);

        // pass in the derived class for proper inheritance
        autograd = std::make_unique<AutogradEngineDebuggerRelease>(
            d_model,
            vocab_size,
            num_heads,
            seq_len,
            batch_size,
            debug);

        // BUFFER CPU/GPU allocation for the auto grad engine
        dl_dz_out_host = (float *)malloc(batch_size * seq_len * vocab_size * sizeof(float));
        cudaMalloc((void **)&dl_dz_out_device, batch_size * seq_len * vocab_size * sizeof(float));

        dataLoader = std::make_unique<DataLoader>(batch_size, data, seq_len, drop_last);

        lm_head = std::make_unique<Linear>(d_model, vocab_size, seq_len, batch_size, num_heads, debug);

        cudaMalloc((void **)&DeviceSoftmaxBLin, batch_size * seq_len * vocab_size * sizeof(float)); // wont something like this reserve GDDR RAM for too long till the liftspan of the object? Yes. -Avash
        cudaMalloc((void **)&DeviceSoftmaxBLout, batch_size * seq_len * vocab_size * sizeof(float));

        cudaMalloc((void **)&outCrossEntropyDevice, batch_size * seq_len * sizeof(float)); // (N) across all of the BT we will have the loss.
        cudaMemset(outCrossEntropyDevice, 0, batch_size * seq_len * sizeof(float));        // make it zero for the things that we are not touching.

        // memory for device for y
        cudaMalloc((void **)&deviceY, batch_size * seq_len * sizeof(int));

        // YOU DO NEED THIS PART BECAUSE ANOTHER KERNEL FOR THE CORSS ENTROPY LOSS WILL BE USING THIS.
        cudaMalloc((void **)&yHotEncodeDeviceOut, vocab_size * seq_len * batch_size * sizeof(float)); // (B, T)
        cudaMemset(yHotEncodeDeviceOut, 0, batch_size * seq_len * vocab_size * sizeof(float));        // make default 0

        outCrossEntropyHost = (float *)malloc(batch_size * seq_len * vocab_size * sizeof(float));

        // -- For testing if the kernel launch for one hot works, we have wrapped two kernels for the cross entropy loss REMOVE FOR PERFORMACE
        outHotEncodeOut = (float *)malloc(batch_size * seq_len * vocab_size * sizeof(float));

        cudaMalloc((void **)&out_h, batch_size * seq_len * vocab_size * sizeof(float));

        cudaMalloc((void **)&dl_dw_out_device, batch_size * d_model * vocab_size * sizeof(float));
        dl_dw_out_host = (float *)malloc(batch_size * d_model * vocab_size * sizeof(float));

        // ------------ for w in lm head --------------

        cudaMalloc((void **)&w_device, d_model * vocab_size * sizeof(float));
        cudaMalloc((void **)&w_out_d, d_model * vocab_size * sizeof(float));

        // ----- For flash attention kernel -------------------
        cudaMalloc((void **)&S_device, seq_len * batch_size * num_heads * head_dim * sizeof(float));
        cudaMalloc((void **)&P_device, batch_size * num_heads * seq_len * seq_len * sizeof(float));
        cudaMalloc((void **)&O_device, batch_size * num_heads * seq_len * head_dim * sizeof(float));

        // same size as P just re-arranged row and cols by the def.
        cudaMalloc((void **)&P_T_device, batch_size * num_heads * seq_len * seq_len * sizeof(float));
        // Shape of value matrix  (B, n_head, T, head_dim)
        cudaMalloc((void **)&V_T_device, batch_size * num_heads * seq_len * head_dim * sizeof(float));

        // output, I know not the most efficient kernel or way but lets get to the end-result here first.
        cudaMalloc((void **)&P_T_device_out, batch_size * num_heads * seq_len * seq_len * sizeof(float));
        cudaMalloc((void **)&V_T_device_out, batch_size * num_heads * seq_len * head_dim * sizeof(float));

        cudaMalloc((void **)&Uncontact_G_Upstream, batch_size * seq_len * num_heads * head_dim * sizeof(float));

        cudaMalloc((void **)&Contact_G_Upstream, batch_size * seq_len * d_model * sizeof(float));

        cudaMalloc((void **)&dV, batch_size * num_heads * seq_len * head_dim * sizeof(float));
        cudaMalloc((void **)&dP, batch_size * num_heads * seq_len * seq_len * sizeof(float));

        cudaMalloc((void **)&ppt, batch_size * num_heads * seq_len * seq_len * sizeof(float));

        // dQ and dK for the QK^T backpropagation
        cudaMalloc((void **)&dQ, batch_size * seq_len * num_heads * head_dim * sizeof(float));
        cudaMalloc((void **)&dK, batch_size * seq_len * num_heads * head_dim * sizeof(float));

        cudaMalloc((void **)&d_score_t, batch_size * num_heads * seq_len * seq_len * sizeof(float));

        cudaMalloc((void **)&WkT, d_model * d_model * sizeof(float));
        cudaMalloc((void **)&WvT, d_model * d_model * sizeof(float));
        cudaMalloc((void **)&WqT, d_model * d_model * sizeof(float));

        cudaMalloc((void **)&Wk, d_model * d_model * sizeof(float));
        cudaMalloc((void **)&WV, d_model * d_model * sizeof(float));
        cudaMalloc((void **)&WQ, d_model * d_model * sizeof(float));

        cudaMalloc((void **)&qUp, batch_size * seq_len * d_model * sizeof(float));
        cudaMalloc((void **)&kUp, batch_size * seq_len * d_model * sizeof(float));
        cudaMalloc((void **)&vUp, batch_size * seq_len * d_model * sizeof(float));

        cudaMalloc((void **)&G_x_hat, batch_size * seq_len * d_model * sizeof(float));

        cudaMalloc((void **)&dqWt, batch_size * seq_len * d_model * sizeof(float));
        cudaMalloc((void **)&dkWt, batch_size * seq_len * d_model * sizeof(float));
        cudaMalloc((void **)&dvWt, batch_size * seq_len * d_model * sizeof(float));

        cudaMalloc((void **)&device_x, batch_size * seq_len * d_model * sizeof(float));

        cudaMalloc((void **)&dbeta, d_model * sizeof(float));
        cudaMalloc((void **)&dgamma, d_model * sizeof(float));
    }

    ~AttentionInterface()
    {
        cudaFree(DeviceSoftmaxBLin);
        cudaFree(DeviceSoftmaxBLout);

        cudaFree(yHotEncodeDeviceOut);
        cudaFree(outCrossEntropyDevice);

        cudaFree(deviceY);

        free(outCrossEntropyHost);

        (outHotEncodeOut != nullptr ? free(outHotEncodeOut) : void());

        cudaFree(dl_dz_out_device);
        free(dl_dz_out_host);

        cudaFree(out_h);

        cudaFree(dl_dw_out_device);
        free(dl_dw_out_host);

        cudaFree(w_out_d);
        cudaFree(w_device);

        // free flash-attention device allocations
        cudaFree(S_device);
        cudaFree(P_device);
        cudaFree(O_device);

        cudaFree(P_T_device);
        cudaFree(V_T_device);

        cudaFree(V_T_device_out);
        cudaFree(P_T_device_out);

        cudaFree(Uncontact_G_Upstream);

        cudaFree(Contact_G_Upstream);

        cudaFree(dV);
        cudaFree(dP);
        cudaFree(ppt);

        cudaFree(dQ);
        cudaFree(dK);

        cudaFree(d_score_t);

        cudaFree(WkT);
        cudaFree(WvT);
        cudaFree(WqT);

        cudaFree(Wk);
        cudaFree(WV);
        cudaFree(WQ);

        cudaFree(qUp);
        cudaFree(kUp);
        cudaFree(vUp);

        cudaFree(dkWt);
        cudaFree(dqWt);
        cudaFree(dkWt);

        cudaFree(G_x_hat);

        cudaFree(device_x);

        cudaFree(dbeta);
        cudaFree(dgamma);
    }

    LinearParams getLmHeadParams()
    {
        return LinearParams{lm_head->getWeight(), lm_head->getBias()};
    }

    void softmaxAcrossProballityCrossEntropyLoss(float *probality, int *y)
    {
        cudaMemcpy(DeviceSoftmaxBLin, probality, batch_size * seq_len * vocab_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceY, y, batch_size * seq_len * sizeof(int), cudaMemcpyHostToDevice);

        // this accounts for whatever we are doing softmax on being greater than 32 i.e warp size.
        // THIS IS SOFTMAX ACROSS the logits (B, T, vocab_size)

        /*
            DeviceSoftmaxBLout: Proballity from logits (predicted)
            yHotEncodeDeviceOut: Actual Y (B, T) -> (B, T, vocab_size)
            outCrossEntropyDevice: CE out flat list
        */
        softmax2D(
            DeviceSoftmaxBLin,
            DeviceSoftmaxBLout,
            batch_size,
            seq_len,
            vocab_size);

        cudaMemcpy(probality, DeviceSoftmaxBLout, batch_size * seq_len * vocab_size * sizeof(float), cudaMemcpyDeviceToHost);

        CrossEntropy(
            DeviceSoftmaxBLout, // out from softmaxed
            deviceY,
            yHotEncodeDeviceOut,
            outCrossEntropyDevice,
            batch_size,
            seq_len,
            vocab_size);

        cudaMemcpy(outCrossEntropyHost, outCrossEntropyDevice, batch_size * seq_len * sizeof(float), cudaMemcpyDeviceToHost);

        cudaMemcpy(outHotEncodeOut, yHotEncodeDeviceOut, batch_size * vocab_size * seq_len * sizeof(float), cudaMemcpyDeviceToHost);

        if (debug)
        {
            //  Here basically every element of the y must have its position encoded based on all of the vocab_size
            // std::cout << "Vocab size of all BT" << std::endl;
            // utils->printLastOneOf3D(probality, batch_size, seq_len, vocab_size);

            // std::cout << "Before one hot encode" << std::endl;
            // utils->printFlatArray2D(y, batch_size, seq_len);

            // // Shape
            // std::cout << " After one hot encode " << std::endl;
            // utils->printFlatArray3D(outHotEncodeOut, batch_size, seq_len, vocab_size);

            // std::cout << "Apply the cross entropy loss" << std::endl;
            // utils->printFlatArray1D(outCrossEntropyHost, seq_len * batch_size);

            // std::cout << "Predicted" << std::endl;
            // DebugBTCFlatArray3D(DeviceSoftmaxBLout, batch_size, seq_len, vocab_size);

            // std::cout << "Actual" << std::endl;
            // DebugBTCFlatArray3D(yHotEncodeDeviceOut, batch_size, seq_len, vocab_size);
        }

        debug = false;
    }

    void train(int epoch)
    {
        Batch batch;

        for (int i = 0; i < epoch; ++i)
        {
            while (!(batch = dataLoader->iter()).empty())
            {
                // std::cout << batch.width << std::endl;
                // utils->printFlatArray2D(batch.x, seq_len, batch.width);

                // Shape (B, T, C) only pointer dependent upon the channel dimension is here
                // And there is an illegal memory access somewhere here.
                float *x = attention->forward(batch);

                // if (debug)
                // {
                //     std::cout << " Before LM head " << std::endl;
                //     this->utils->printFlatArray3D(x, batch_size, seq_len, d_model);
                // }

                float *prob = lm_head->forward(x); // Shape (B, T, vocab_size) x is not changed here.

                modelParamaters.w_host = lm_head->getWeight();

                // if (debug)
                // {
                //     std::cout << " After LM head " << std::endl;
                //     this->utils->printLastOneOf3D(prob, batch_size, seq_len, vocab_size);
                // }

                softmaxAcrossProballityCrossEntropyLoss(prob, batch.y);

                // if (debug)
                // {
                //     std::cout << " After sfotmax last two dimension " << std::endl;
                //     this->utils->printLastOneOf3D(prob, batch_size, seq_len, vocab_size);
                // }

                // ----------- Lets gather overall paramaters here ---------- //
                modelParamaters.attention_head = attention->getParamaters();
                modelParamaters.lm_head = LinearParams{lm_head->getWeight(), lm_head->getBias()};
                modelParamaters.L = outCrossEntropyDevice;        // CE Out
                modelParamaters.y_actual = yHotEncodeDeviceOut;   // (B, T, vocab_size) from actual
                modelParamaters.y_predicted = DeviceSoftmaxBLout; // (B, T, vocab_size) to predicted proballity

                modelParamaters.dl_dz_out_device = dl_dz_out_device;
                modelParamaters.dl_dz_out_host = dl_dz_out_host;
                modelParamaters.h = x;                                   // this h is the output of lm head Shape(B, T, vocab_size)
                modelParamaters.device_h = attention->BorrowBTCDevice(); // (B, T, d_model) on device
                modelParamaters.device_out_h = out_h;

                // for dl_dw = delta h^T derived in flashback.md

                modelParamaters.dl_dw_device = dl_dw_out_device;
                modelParamaters.dl_dw_host = dl_dw_out_host;

                // these are just borrowed pointers
                modelParamaters.w_device = w_device;
                modelParamaters.wt_out_d = w_out_d;

                modelParamaters.P_T_device = P_T_device;
                modelParamaters.V_T_device = V_T_device;

                modelParamaters.P_T_device_out = P_T_device_out;
                modelParamaters.V_T_device_out = V_T_device_out;

                // GDDR RAM
                modelParamaters.Uncontact_G_Upstream = Uncontact_G_Upstream;
                modelParamaters.Contact_G_Upstream = Contact_G_Upstream;

                modelParamaters.dV = dV;
                modelParamaters.dP = dP;

                modelParamaters.ppt = ppt;

                modelParamaters.dQ = dQ;
                modelParamaters.dK = dK;

                modelParamaters.d_score_t = d_score_t;

                modelParamaters.WkT = WkT;
                modelParamaters.WvT = WvT;
                modelParamaters.WqT = WqT;

                modelParamaters.Wk = Wk;
                modelParamaters.WV = WV;
                modelParamaters.WQ = WQ;

                modelParamaters.qUp = qUp;
                modelParamaters.kUp = kUp;
                modelParamaters.vUp = vUp;

                modelParamaters.dqWt = dqWt;
                modelParamaters.dkWt = dkWt;
                modelParamaters.dvWt = dvWt;

                modelParamaters.G_x_hat = G_x_hat; // net G_hat from the derivation

                modelParamaters.device_x = device_x;

                modelParamaters.debeta = dbeta;
                modelParamaters.dgamma = dgamma;

                // because the backprops needs to be done for each epoch.
                // we need to keep in mind that the things hurting performace like cuda malloc and everything declared
                // inside of the constructor of autograd engine is costly.
                // there is tradeoff between making things modular and fusing everything together.
                // Lets create a buffer for CPU/GPU memory in this class so that we dont overload the system and free it when the object is destroyed.
                autograd->backprop(modelParamaters);
                debug = false;
            }
            dataLoader->resetIterator(); // just the weird logic that I wrote.
        }
    }
};
