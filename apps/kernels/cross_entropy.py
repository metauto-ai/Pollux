import math
import torch
import triton
import triton.language as tl
from typing import Optional
import torch.distributed as dist
from kernels.configure import MAX_FUSED_SIZE, calculate_settings

# forward and backward pass in single kernel, no need to save anything.

@triton.jit
def _cross_entropy_kernel(
    logits_ptr,
    target_ptr,
    loss_ptr,
    dlogits_ptr,
    vocab_size,
    stride,
    n_rows,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    row = pid * vocab_size
    n_blocks = (vocab_size + BLOCK_SIZE - 1) // BLOCK_SIZE

    # cross entropy loss(l) = -log(softmax(logits))
    # l = -log(exp(logits[target]) / sum(exp(logits)))
    # l = log(sum(exp(logits))) - logits[target]
    # so here logits[target] is of size 1 and logits is of size vocab_size, reduced to sum over the row
    # we want the exp(logits) to be stable, so we subtract the maximum value(in the row) from each logit
    # re-formed formula => l = -log(softmax(logits(T)))
    # l = -log(exp(T - maxi)/sum(exp(logits - maxi)))
    # l = log(sum(exp(logits - maxi))) - (T - maxi)
    # l = [log(sum(exp(logits - maxi))) + maxi] - T
    #     |__________1st pass_______________|  
    #     logsumexp - T --> 2nd pass

    # the maximum of data we can load into mem is 65536, so for larger vocab size we need to split he row into blocks and perform ops
    # lets say we have 128k in a row, we split into two blocks of 64k, iterate through each block

    # so for the first pass we iterate over the individual blocks, calculate sums for each block, calculate the global maximum and finally calculate the logsumexp
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < vocab_size
    logsumexp = 0.0
    sumexp = 0.0
    global_max = float('-inf')
    # for i in range(n_blocks):
    #     block_start = row + i * BLOCK_SIZE
    #     row_block = tl.load(logits_ptr + block_start + cols, mask = mask)
    #     maxi = tl.max(row_block)
    #     global_maxi = tl.max(maxi, global_maxi)
    
    # for i in range(n_blocks):
    #     block_start = row + i * BLOCK_SIZE
    #     row_block = tl.load(logits_ptr + block_start + cols, mask = mask)
    #     row_block = row_block - global_maxi
    #     row_block = tl.exp(row_block)
    #     logsumexp += tl.sum(row_block)
    # so in th above method, we are loading the row twice, so total no of reads will double, instead online softmax introduced a method to calculate the logsumexp in a single pass

    # online softmax => for logsumexp
    # block_max = max(global_max, max(row_block))
    # sum = 0
    # sum += sum*exp(previous_block_max - block_max) + exp(row_block - block_max)
    for i in range(n_blocks):
        block_start = row + i * BLOCK_SIZE
        row_block = tl.load(logits_ptr + block_start + cols, mask = mask)
        block_max = tl.max(row_block)
        prev_max = global_max
        global_max = block_max if block_max > global_max else global_max
        sumexp += sumexp * tl.exp(prev_max - global_max) + tl.sum(tl.exp(row_block - global_max))

    # for the second pass we just calculate the loss
    logsumexp = tl.log(sumexp) + global_max
    target = tl.load(target_ptr + pid)
    target_prob = tl.load(logits_ptr + row + target)
    loss = logsumexp - target_prob
    tl.store(loss_ptr + pid, loss)

    # now onto backward pass
    # l = (log(sum(exp(logits - maxi))) + maxi - T).mean(), so dl/dx = softmax(logits)/n
    # dx = softmax(x)/n 
    # for target index dx -= 1/n

    # first backward pass --> we calculate softmax with the online normalizer cal
    # for online softmax, we reuse the sumexp, global_max
    # softmax = exp(logits - global_max)/sumexp
    for i in range(n_blocks):
        block_start = row + i * BLOCK_SIZE
        row_block = tl.load(logits_ptr + block_start + cols, mask = mask)
        row_block = row_block - global_max
        row_block = tl.exp(row_block)
        row_block = (row_block / sumexp) / n_rows
        tl.store(dlogits_ptr + block_start + cols, row_block, mask = mask)
    
    # second backward pass --> we subtract 1/n from the target index
    dlogits_target = tl.load(dlogits_ptr + row + target)
    dlogits_target -= (1.0 / n_rows)
    tl.store(dlogits_ptr + row + target, dlogits_target)
    
    # bazzinga

class FastCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target):
        n_rows, vocab_size = logits.shape
        
        if not logits.is_contiguous():
            logits = logits.contiguous()
        if not target.is_contiguous():
            target = target.contiguous()
        
        losses = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        dlogits = torch.empty((n_rows, vocab_size), dtype=torch.float32, device=logits.device)

        BLOCK_SIZE, num_warps = calculate_settings(vocab_size)
        
        BLOCK_SIZE = min(MAX_FUSED_SIZE, BLOCK_SIZE)
        
        _cross_entropy_kernel[(n_rows,)](
            logits,
            target,
            losses,
            dlogits,
            vocab_size,
            vocab_size,
            n_rows,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps
        )
        
        ctx.save_for_backward(logits, dlogits)
        return losses.mean()

    @staticmethod
    def backward(ctx, grad_output):
        logits, dlogits = ctx.saved_tensors
        
        return dlogits, None

def fast_cross_entropy(logits, target):
    """
    Numerically stable cross entropy loss optimized for DDP/FSDP.
    
    Args:
        logits: Input tensor of shape [batch_size, vocab_size]
        target: Target indices of shape [batch_size]
        
    Returns:
        Scalar loss value
    """
    return FastCrossEntropy.apply(logits, target)
