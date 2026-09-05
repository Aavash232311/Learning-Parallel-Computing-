import torch
import numpy as np
from ml_components.grad import load_tensor

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

def ReaderFlashAttention(
        batch_size: int,
        seq_len: int,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        head_dim: int
):
    # batch_size * num_heads * seq_len * seq_len = P
    # batch_size * num_heads * seq_len * head_dim = V
    P = load_tensor('./src/cache/cpp_out/P.bin', shape=(batch_size, num_heads, seq_len, seq_len), dtype=np.float32).to(device)
    V = load_tensor('./src/cache/cpp_out/V.bin', shape=(batch_size, num_heads, seq_len, head_dim), dtype=np.float32).to(device)

    # Shape is same, value is different after the transpose operation takes place
    PT = load_tensor('./src/cache/cpp_out/pt.bin', shape=(batch_size, num_heads, seq_len, seq_len), dtype=np.float32).to(device)
    VT = load_tensor('./src/cache/cpp_out/vt.bin', shape=(batch_size, num_heads, head_dim, seq_len), dtype=np.float32).to(device)

    G_unc = load_tensor('./src/cache/cpp_out/G_uncontact.bin', shape=(batch_size, num_heads, seq_len, head_dim), dtype=np.float32).to(device)
    dl_dh = load_tensor('./src/cache/cpp_out/dl_dh.bin', shape=(batch_size, seq_len, d_model), dtype=np.float32).to(
        device)


    dp = load_tensor("./src/cache/cpp_out/dp.bin",
                     shape=(batch_size, num_heads, seq_len, seq_len),
                     dtype=np.float32).to(device)

    dV = load_tensor("./src/cache/cpp_out/dv.bin",
                     shape=(batch_size, num_heads, seq_len, head_dim),
                     dtype=np.float32).to(device)

    softmax_upstream = load_tensor("./src/cache/cpp_out/softmax_upstream.bin",
                                   shape=(batch_size, num_heads, seq_len, seq_len),
                                   dtype=np.float32).to(device)

    dQ = load_tensor("./src/cache/cpp_out/dq.bin",
                     shape=(batch_size, num_heads, seq_len, head_dim),
                     dtype=np.float32
                     ).to(device)

    # load K and Q cache in the debugger
    k = load_tensor("./src/cache/cpp_out/k.bin",
                    shape=(batch_size, num_heads, seq_len, head_dim), # (batch_size, num_head, seq_len, head_dim)
                    dtype=np.float32
                    ).to(device)
    q = load_tensor("./src/cache/cpp_out/q.bin",
                    shape=(batch_size, num_heads, seq_len, head_dim), # (batch_size, num_head, seq_len, head_dim)
                    dtype=np.float32
                    ).to(device)

    d_score_t = load_tensor("./src/cache/cpp_out/d_score_t.bin",
                        shape=(batch_size, num_heads, seq_len, seq_len),
                        dtype=np.float32
                    ).to(device)

    dK = load_tensor("./src/cache/cpp_out/dk.bin",
                     shape=(batch_size, num_heads, seq_len, head_dim),
                     dtype=np.float32
                     ).to(device)

    # LayerNorm's mean and std dev cache, Note: just to check if the data is healthy
    mean_cache = load_tensor("./src/cache/cpp_out/mean_cache.bin", shape=(batch_size, seq_len), dtype=np.float32).to(device)
    std_dev_cache = load_tensor("./src/cache/cpp_out/std_dev_cache.bin", shape=(batch_size, seq_len),
                             dtype=np.float32).to(device)


    wqt = load_tensor("./src/cache/cpp_out/wqt.bin",
                      shape=(d_model, d_model),
                      dtype=np.float32
                      ).to(device)

    wkt = load_tensor("./src/cache/cpp_out/wkt.bin",
                      shape=(d_model, d_model),
                      dtype=np.float32
                      ).to(device)

    wvt = load_tensor("./src/cache/cpp_out/wvt.bin",
                      shape=(d_model, d_model),
                      dtype=np.float32
                      ).to(device)

    wq = load_tensor("./src/cache/cpp_out/wq.bin",
                      shape=(d_model, d_model),
                      dtype=np.float32
                      ).to(device)
    wk = load_tensor("./src/cache/cpp_out/wk.bin",
                      shape=(d_model, d_model),
                      dtype=np.float32
                      ).to(device)
    wv = load_tensor("./src/cache/cpp_out/wv.bin",
                      shape=(d_model, d_model),
                      dtype=np.float32
                      ).to(device)

    upq = load_tensor("./src/cache/cpp_out/upQ.bin",
                      shape=(batch_size, seq_len, d_model),
                      dtype=np.float32
                      ).to(device)

    upk = load_tensor("./src/cache/cpp_out/upK.bin",
                      shape=(batch_size, seq_len, d_model),
                      dtype=np.float32
                      ).to(device)

    upv = load_tensor("./src/cache/cpp_out/upV.bin",
                      shape=(batch_size, seq_len, d_model),
                      dtype=np.float32
                      ).to(device)

    G_x_hat = load_tensor("./src/cache/cpp_out/G_x_hat.bin",
                          shape=(batch_size, seq_len, d_model),
                          dtype=np.float32
                          ).to(device)

    layer_norm_gamma = load_tensor("./src/cache/cpp_out/gamma_host.bin",
                        shape=(batch_size, 1, 1),
                        dtype=np.float32
                        ).to(device)
    x = load_tensor("./src/cache/cpp_out/embedding.bin",
                    shape=(batch_size, seq_len, d_model),
                    dtype=np.float32).to(device)
    return P, V, PT, VT, G_unc, dl_dh, dp, dV, softmax_upstream, dQ, k, q, d_score_t, dK, wqt, wkt, wvt, wq, wk, wv, upq, upk, upv, G_x_hat, layer_norm_gamma, mean_cache, std_dev_cache, x

def Reader(
        batch_size: int,
        seq_len: int,
        vocab_size: int,
        d_model: int
):
    delta = load_tensor('./src/cache/cpp_out/delta.bin', shape=(batch_size, seq_len, vocab_size), dtype=np.float32).to(device)
    y_predicted = load_tensor('./src/cache/cpp_out/y_prediced.bin', shape=(batch_size, seq_len, vocab_size), dtype=np.float32).to(device)
    y_actual = load_tensor('./src/cache/cpp_out/y_actual.bin', shape=(batch_size, seq_len, vocab_size), dtype=np.float32).to(device)
    h = load_tensor('./src/cache/cpp_out/h.bin', shape=(batch_size, seq_len, d_model), dtype=np.float32).to(device)
    # (B, C, vocab_size)
    dl_dw_kernel = load_tensor('./src/cache/cpp_out/dl_dw.bin', shape=(batch_size, d_model, vocab_size), dtype=np.float32).to(device)
    h_t = load_tensor('./src/cache/cpp_out/h_t.bin', shape=(batch_size, d_model, seq_len), dtype=np.float32).to(device)

    wt = load_tensor('./src/cache/cpp_out/wt.bin', shape=(vocab_size, d_model), dtype=np.float32).to(device)
    w = load_tensor('./src/cache/cpp_out/w.bin', shape=(d_model, vocab_size), dtype=np.float32).to(device)

    dl_dh = load_tensor('./src/cache/cpp_out/dl_dh.bin', shape=(batch_size, seq_len, d_model), dtype=np.float32).to(device)


    return delta, y_predicted, y_actual, h, dl_dw_kernel, h_t, wt, w, dl_dh

