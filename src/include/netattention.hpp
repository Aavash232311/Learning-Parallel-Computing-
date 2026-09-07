#pragma once
#include "attention_params.hpp"
#include "linear.hpp"

struct NetAttentionParamaters
{
    AttentionParamaters attention_head;
    LinearParams lm_head;

    float *L; // loss from the cross entropy loss starting of the backpropagation
    // so the above struct contains a pointer reference for CPU memory but we need to make a buffer for gpu memory right here.

    // --------- Variables needed for upstream gradient --------------

    // MAKE SURE THAT THSE ARE VARIABLES FROM THE DEVICE
    float *y_actual;
    float *y_predicted;
    float *dl_dz_out_device;
    float *dl_dz_out_host; // upstream gradient dl/dz

    float *dl_dw_device;
    float *dl_dw_host;

    // -------------- Linear Layer LM head paramaters ------------
    float *h; // (B, T, d_model) output from the attention head
    float *device_h;
    float *device_out_h; // (B, C, T) transpose head

    float *w_host;
    float *w_device;
    float *wt_out_d; // copy to CPU if the debugger is on and see.

    float *dl_dh_host;
    float *dl_dh_device;
    float *dl_dh_out_d;
};

struct FlashAttentionPointers : NetAttentionParamaters
{
    // normal opreation like transpose is checked by the python debugger.
    // we will create a temporary variable when we want to release these.
    float *P_T_device;
    float *V_T_device;

    // wewe P_T and V_T to be output such that we can consume the passed arr
    float *P_T_device_out;
    float *V_T_device_out;

    float *Uncontact_G_Upstream;
    float *Contact_G_Upstream;

    float *dV;
    float *dP;

    float *ppt;

    float *dQ;
    float *dK;

    float *d_score_t;

    // weights of Q, K and V
    // These are in device from linear

    float *WkT;
    float *WvT;
    float *WqT; // shape in LM head and shape of in attention head is not the same I got confused

    float *Wk;
    float *WV;
    float *WQ;

    // Correctness first optimization layer, already very very complicated

    // G_x_hat

    // compact values pointers
    float *qUp;
    float *kUp;
    float *vUp;

    // resultant value of
    // dV wT, dQ wT, dK wT you get it.

    float *dqWt;
    float *dkWt;
    float *dvWt;

    // G_x_hat
    float *G_x_hat;

    // x after embedding is shape (B, T,C)
    // which we need for the layer norm backpropagation kernel
    float *device_x; // wont want to write something as generic as dx

    float *debeta;
    float *dgamma;
};