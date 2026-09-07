import os
import torch
from pathlib import Path


from rnd.release_embeddings import release_token_embeddings
from debug.debug_autograd import debug_autograd
from debug.debug_embeddings import verify_embeddings
from debug.debug_flash_attention import DebugFlashAttention

from binary_reader.hyperparamaters import read_hyperparamaters

''' Automated debugging script for CUDA kernel in attention.cpp
    File based dump verification.
    Never trust the debugger too much, this is for basic mathematical verification (for specific operations) only.
    Use local print and this combined :)
    
    THIS IS JUST MY ROUGH WORK NOTHING ELSE A SMALL SANITY CHECK
    
    Wont notice things like infinitely large values in a kernel sometimes,
    just focuses on mathematical operations. 
 '''
# Make sure we are able to read C++ project directory from here
os.chdir(Path.cwd().parent)
# Read release hyperparameters, compiles the C++ and releases the hyperparamaters.
d_model, vocab_size, batch_size, seq_len, num_heads = read_hyperparamaters()

token_emb = release_token_embeddings(d_model=d_model, seq_len=seq_len, vocab_size=vocab_size, batch_size=batch_size)

# After hyperparamaters are loaded, we need to load those binary that the C++ releases
os.system("nvcc -DDEBUG -DDRUN  src/main.cpp src/kernel/utils.cu src/forward/Kernel/layer_norm.cu src/forward/Kernel/embedding.cu src/forward/Kernel/linear.cu src/forward/Kernel/attention_head.cu src/forward/Kernel/interface.cu src/backpropagation/Kernel/interface_back.cu src/backpropagation/Kernel/flash_attention.cu -o src/bin/attention")
os.system("./src/bin/attention")

print("\n")
print('*' * 60)
print("Forward pass ")
print('*' * 60)

# check if embedding component of the transformer is okay.
verify_embeddings(d_model=d_model, seq_len=seq_len, batch_size=batch_size, vocab_size=vocab_size, num_heads=num_heads, token_embeddings=token_emb)

torch.set_printoptions(precision=4)

input_ids = torch.randint(0, vocab_size, (seq_len, batch_size))
input_ids.to(torch.int32).numpy().tofile("./src/cache/pytorch_out/input_ids.bin")


G = debug_autograd(d_model=d_model, seq_len=seq_len, batch_size=batch_size, vocab_size=vocab_size, num_heads=num_heads)


flash_attention_debugger = DebugFlashAttention(d_model=d_model, seq_len=seq_len, batch_size=batch_size, vocab_size=vocab_size, num_heads=num_heads, head_dim=d_model // num_heads, dl_dw=G)
flash_attention_debugger.victor_tango()

# Also note if somewhere in forward pass or early execution sequence if something is
# corrupted then executing this will take more time than usual