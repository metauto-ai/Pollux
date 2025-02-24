import math
import torch
import triton
import triton.language as tl
from typing import Optional
import torch.distributed as dist
import numpy as np

HAS_TMA_DESC = "nv_tma_desc_type" in dir(tl)

class TmaAutoTuneHelper:
    # duck typing wrapper to implement the same interface as TmaDescKernelParam
    class KernelParamWrapper:
        def __init__(self, desc):
            self.desc = desc

        def tma_desc_cpu_ptr(self):
            return self.desc.data_ptr()

    TMA_SIZE = 128

    def __init__(self):
        self.fill_1d_tma_descriptor_inner = triton.runtime.driver.active.utils.fill_1d_tma_descriptor
        self.fill_2d_tma_descriptor_inner = triton.runtime.driver.active.utils.fill_2d_tma_descriptor
        if HAS_TMA_DESC:
            self.descriptors = {}
        else:
            self.cuda_descriptors = {}

    def init_tma_descriptor(self, name):
        if HAS_TMA_DESC:
            self.descriptors[name] = torch.empty(TmaAutoTuneHelper.TMA_SIZE, device="cpu", dtype=torch.int8)
        else:
            self.cuda_descriptors[name] = torch.empty(TmaAutoTuneHelper.TMA_SIZE, device="cuda", dtype=torch.int8)

    def fill_2d_tma_descriptor(self, name, ptr, dim1, dim0, block_dim1, block_dim0, element_size):
        if HAS_TMA_DESC:
            desc = self.descriptors[name]
            self.fill_2d_tma_descriptor_inner(
                ptr,
                dim1,
                dim0,
                block_dim1,
                block_dim0,
                element_size,
                desc.data_ptr()
            )
        else:
            desc = self.cuda_descriptors[name]
            self.fill_2d_tma_descriptor_inner(
                ptr,
                dim1,
                dim0,
                block_dim1,
                block_dim0,
                element_size,
                desc.data_ptr()
            )

    def fill_1d_tma_descriptor(self, name, ptr, dim, block_dim, element_size):
        if HAS_TMA_DESC:
            desc = self.descriptors[name]
            self.fill_1d_tma_descriptor_inner(
                ptr,
                dim,
                block_dim,
                element_size,
                desc.data_ptr()
            )
        else:
            desc = self.cuda_descriptors[name]
            self.fill_1d_tma_descriptor_inner(
                ptr,
                dim,
                block_dim,
                element_size,
                desc.data_ptr()
            )

    def get_tma_descriptor_kernel_param(self, name):
        if HAS_TMA_DESC:
            return TmaAutoTuneHelper.KernelParamWrapper(self.descriptors[name])
        else:
            return self.cuda_descriptors[name]

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 65536}, num_stages=3, num_warps=32, num_consumer_groups=4, num_buffers_warp_spec=3,),
        triton.Config({"BLOCK_SIZE": 128000}, num_stages=3, num_warps=64, num_consumer_groups=8, num_buffers_warp_spec=4,),
    ],
    key=['vocab_size'],
)
@triton.jit
def _cross_entropy_kernel(
    logits_desc_ptr,  # TMA descriptor for logits
    target_ptr,
    loss_desc_ptr,    # TMA descriptor for losses
    vocab_size,
    stride,
    n_rows,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    row = (pid * vocab_size).to(tl.int32)
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

    # the maximum of data we can load into mem is 65536, so for larger vocab size we need to split the row into blocks and perform ops
    # lets say we have 128k in a row, we split into two blocks of 64k, iterate through each block

    # Precompute mask offsets outside the loop for efficiency
    cols = tl.arange(0, BLOCK_SIZE)
    
    # Initialize accumulators
    global_max = float('-inf')
    sumexp = 0.0
    
    # First pass: find max
    for i in range(n_blocks):
        block_start = (i * BLOCK_SIZE).to(tl.int32)
        block_size = min(BLOCK_SIZE, vocab_size - block_start).to(tl.int32)
        
        # Load block using TMA with proper shape handling
        logits_block = tl._experimental_descriptor_load(
            logits_desc_ptr,
            [pid, block_start],  
            [1, block_size],     
            dtype=tl.float32
        )
        
        # Update max, masking out invalid elements
        mask = offsets < block_size
        block_max = tl.max(logits_block, mask=mask)
        global_max = tl.maximum(global_max, block_max)
    
    # Second pass: compute sum of exponentials
    for i in range(n_blocks):
        block_start = (i * BLOCK_SIZE).to(tl.int32)
        block_size = min(BLOCK_SIZE, vocab_size - block_start).to(tl.int32)
        
        # Load block using TMA
        logits_block = tl._experimental_descriptor_load(
            logits_desc_ptr,
            [pid, block_start],
            [1, block_size],
            dtype=tl.float32
        )
        
        # Compute exp(x - max) and sum, with masking
        mask = offsets < block_size
        exp_block = tl.exp(logits_block - global_max)
        sumexp += tl.sum(exp_block, mask=mask)
    
    # Load target index
    target_idx = tl.load(target_ptr + pid)
    
    # Load target logit using TMA
    target_logit = tl._experimental_descriptor_load(
        logits_desc_ptr,
        [pid, target_idx],
        [1, 1],
        dtype=tl.float32
    )
    
    # Compute final loss
    loss = tl.log(sumexp) + global_max - target_logit
    
    # Store result using TMA
    tl._experimental_descriptor_store(
        loss_desc_ptr,
        loss,
        [pid],
        [1],
        dtype=tl.float32
    )

    # now onto backward pass
    # l = (log(sum(exp(logits - maxi))) + maxi - T).mean(), so dl/dx = softmax(logits)/n
    # dx = softmax(x)/n 
    # for target index dx -= 1/n

    # first backward pass --> we calculate softmax with the online normalizer cal
    # for online softmax, we reuse the sumexp, global_max
    # softmax = exp(logits - global_max)/sumexp
    for i in range(n_blocks):
        block_start = (i * BLOCK_SIZE).to(tl.int32)
        block_size = min(BLOCK_SIZE, vocab_size - block_start).to(tl.int32)
        
        # Load logits block using TMA
        logits_block = tl._experimental_descriptor_load(
            logits_desc_ptr,
            [pid, block_start],
            [1, block_size],
            dtype=tl.float32
        )
        logits_block = logits_block - global_max
        logits_block = tl.exp(logits_block)
        logits_block = (logits_block / sumexp) / n_rows
        # Store gradients using TMA
        tl._experimental_descriptor_store(
            logits_desc_ptr,
            logits_block,
            [pid, block_start],
            [1, block_size],
        )
    
    # second backward pass --> we subtract 1/n from the target index
    dlogits_target = tl._experimental_descriptor_load(
        logits_desc_ptr,
        [pid, target_idx],
        [1, 1],
        dtype=tl.float32
    )
    dlogits_target -= (1.0 / n_rows)
    tl._experimental_descriptor_store(
        logits_desc_ptr,
        dlogits_target,
        [pid, target_idx],
        [1, 1],
    )

class FastCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target):
        n_rows, vocab_size = logits.shape
        
        if not logits.is_contiguous():
            logits = logits.contiguous()
        if not target.is_contiguous():
            target = target.contiguous()
        
        # Create output tensor
        losses = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        
        # Initialize TMA helper
        desc_helper = TmaAutoTuneHelper()
        
        # Initialize descriptors
        desc_helper.init_tma_descriptor("logits")
        desc_helper.init_tma_descriptor("losses")
        
        # Calculate minimum block size for TMA (32 bytes)
        min_block_size = max(32 // logits.element_size(), 8)  # At least 8 elements
        
        # Fill descriptors with appropriate block sizes
        desc_helper.fill_2d_tma_descriptor(
            "logits",
            logits.data_ptr(),
            n_rows,
            vocab_size,
            1,                # block_dim1: process 1 row at a time
            min_block_size,   # block_dim0: ensure at least 32 bytes
            logits.element_size()
        )
        
        desc_helper.fill_1d_tma_descriptor(
            "losses",
            losses.data_ptr(),
            n_rows,
            min_block_size,   # block_dim: ensure at least 32 bytes
            losses.element_size()
        )
        
        # Get descriptor parameters
        logits_desc = desc_helper.get_tma_descriptor_kernel_param("logits")
        losses_desc = desc_helper.get_tma_descriptor_kernel_param("losses")
        
        # Launch kernel
        _cross_entropy_kernel[(n_rows,)](
            logits_desc,
            target,
            losses_desc,
            vocab_size,
            vocab_size,  # stride
            n_rows
        )
        
        # Save for backward
        ctx.save_for_backward(logits)
        ctx.target = target
        
        return losses.mean()

    @staticmethod
    def backward(ctx, grad_output):
        logits, = ctx.saved_tensors
        target = ctx.target
        grad = torch.zeros_like(logits)
        
        # Compute gradients using grad_mul kernel
        grad_mul[(logits.numel(),)](
            logits.data_ptr(),
            logits.stride(-2),
            grad_output,
            logits.shape[-1],
            BLOCK_SIZE=256
        )
        
        return grad, None

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

@triton.jit
def grad_mul(
    X_ptr,
    X_stride,
    grad_output_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)

    X_ptr += program_id * X_stride

    grad_output = tl.load(grad_output_ptr)

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols)
        tl.store(X_ptr + X_offsets, X_block * grad_output, mask=X_offsets < n_cols)
