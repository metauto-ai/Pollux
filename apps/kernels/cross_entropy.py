import math
import torch
import triton
import triton.language as tl
from typing import Optional
import torch.distributed as dist

MAX_FUSED_SIZE : int = 65536
from triton.language.extra.libdevice import tanh

@triton.jit
def _fused_cross_entropy_kernel(
    logits_ptr,
    target_ptr,
    loss_ptr,
    vocab_size,
    stride,
    BLOCK_SIZE: tl.constexpr,
    DO_SOFTCAPPING: tl.constexpr,
    SOFTCAP: tl.constexpr
):
    pid = tl.program_id(0)
    
    row_offset = pid * stride
    
    y = tl.load(target_ptr + pid).to(tl.int32)
    
    if y < 0 or y >= vocab_size:
        for i in range(0, vocab_size, BLOCK_SIZE):
            offs = i + tl.arange(0, BLOCK_SIZE)
            mask = offs < vocab_size
            tl.store(logits_ptr + row_offset + offs, 0.0, mask=mask)
        return
    
    row_max = float('-inf')
    
    for i in range(0, vocab_size, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < vocab_size
        x = tl.load(logits_ptr + row_offset + offs, mask=mask, other=float('-inf'))
        if DO_SOFTCAPPING:
            x = SOFTCAP * tanh(x / SOFTCAP)
        row_max = tl.maximum(row_max, tl.max(x, axis=0))
    
    sum_exp = 0.0
    label_logit = tl.load(logits_ptr + row_offset + y)
    if DO_SOFTCAPPING:
        label_logit = SOFTCAP * tanh(label_logit / SOFTCAP)
    
    for i in range(0, vocab_size, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < vocab_size
        x = tl.load(logits_ptr + row_offset + offs, mask=mask, other=float('-inf'))
        if DO_SOFTCAPPING:
            partial = tanh(x / SOFTCAP)
            x = SOFTCAP * partial
        
        exp_x = tl.exp(x - row_max)
        sum_exp += tl.sum(exp_x, axis=0)
        
        grad = exp_x / (sum_exp + 1e-6)
        grad = tl.where(offs == y, grad - 1.0, grad)
        if DO_SOFTCAPPING:
            grad = grad * (1.0 - partial * partial)
        
        tl.store(logits_ptr + row_offset + offs, grad, mask=mask)
    
    loss = tl.log(sum_exp + 1e-6) + row_max - label_logit
    tl.store(loss_ptr + pid, loss)

class FastCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target, softcap_thresh=0.0):
        n_rows, vocab_size = logits.shape
        
        if not logits.is_contiguous():
            logits = logits.contiguous()
        if not target.is_contiguous():
            target = target.contiguous()
            
        losses = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(vocab_size))
        
        _fused_cross_entropy_kernel[(n_rows,)](
            logits,
            target,
            losses,
            vocab_size,
            vocab_size, 
            BLOCK_SIZE=BLOCK_SIZE,
            DO_SOFTCAPPING=bool(softcap_thresh != 0),
            SOFTCAP=softcap_thresh,
            num_warps=4
        )
        
        ctx.save_for_backward(logits)
        return losses.mean()

    @staticmethod
    def backward(ctx, grad_output):
        logits, = ctx.saved_tensors
        
        if grad_output.ndim == 0:
            grad_output = grad_output.expand(logits.size(0))
            
        return logits * grad_output.view(-1, 1), None, None

def fast_cross_entropy(logits, target, softcap_thresh=0.0):
    """
    Memory-efficient cross entropy loss that works with DDP/FSDP.
    Computes gradients in-place during forward pass.
    
    Args:
        logits: Input tensor of shape [batch_size, vocab_size]
        target: Target indices of shape [batch_size]
        softcap_thresh: Optional threshold for gradient capping
        
    Returns:
        Scalar loss value
    """
    return FastCrossEntropy.apply(logits, target, softcap_thresh)
