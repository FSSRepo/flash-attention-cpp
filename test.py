import time
import torch
import torch.nn.functional as F

bz = 1
seq_len = 4096
dims = 128
n_heads = 32

q = torch.randn(bz, n_heads, 512, dims, dtype=torch.float16).cuda()
k = torch.randn(bz, n_heads, seq_len, dims, dtype=torch.float16).cuda()
v = torch.randn(bz, n_heads, seq_len, dims, dtype=torch.float16).cuda()

dropout_rate = 0.0
num_trials = 10

torch.cuda.synchronize()
start = time.time()
for i in range(num_trials):
    attn = q @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    attn = F.dropout(attn, p=dropout_rate, training=True)
    x = (attn @ v).transpose(1, 2)  # .reshape(bz, seq_len, n_heads*dims)
torch.cuda.synchronize()
end = time.time()
print('Standard attention took {} ms for {} trials'.format((end - start)*1000, num_trials))

with torch.backends.cuda.sdp_kernel(
    enable_flash=True, enable_math=False, enable_mem_efficient=False
):
    torch.cuda.synchronize()
    start = time.time()
    for i in range(num_trials):
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_rate)
    torch.cuda.synchronize()
    end = time.time()
    print('Flash attention took {} ms for {} trials'.format((end - start)*1000, num_trials))