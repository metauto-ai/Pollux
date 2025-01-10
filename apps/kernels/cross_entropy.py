import math
import torch
import triton
import triton.language as tl
from torch.amp import custom_fwd, custom_bwd
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import torch.distributed as dist

@dataclass
class _CrossEntropyConfig:
    BLOCK_SIZE: int = 4096
    NUM_WARPS: int = 4
    NUM_STAGES: int = 2

def _get_optimal_config(vocab_size: int) -> _CrossEntropyConfig:
    if vocab_size <= 4096:
        return _CrossEntropyConfig(BLOCK_SIZE=1024, NUM_WARPS=4, NUM_STAGES=2)
    elif vocab_size <= 32768:
        return _CrossEntropyConfig(BLOCK_SIZE=2048, NUM_WARPS=8, NUM_STAGES=2)
    else:
        return _CrossEntropyConfig(BLOCK_SIZE=4096, NUM_WARPS=8, NUM_STAGES=3)

@triton.jit
def _softcap(x, thresh):
    diff = x - thresh
    return tl.where(x > thresh, thresh + tl.log(1.0 + diff), x)

@triton.jit
def _cross_entropy_fwd_kernel(
    logits,
    target,
    output,
    batch_size,
    vocab_size,
    stride,
    epsilon,
    softcap_thresh,
    BLOCK_SIZE: tl.constexpr,):

    pid = tl.program_id(0)
    if pid >= batch_size:
        return

    row_start = pid * stride
    max_val = float("-inf")
    exp_sum = 0.0
    log_sum = 0.0

    # find max
    for block_start in range(0, vocab_size, BLOCK_SIZE):
        offs = tl.arange(0, BLOCK_SIZE)
        mask = (block_start + offs) < vocab_size
        block = tl.load(logits + row_start + block_start + offs, mask=mask)
        block = _softcap(block, softcap_thresh)
        block_max = tl.max(block, axis=0)
        max_val = max(max_val, block_max)

    # compute exp sum and log sum
    for block_start in range(0, vocab_size, BLOCK_SIZE):
        offs = tl.arange(0, BLOCK_SIZE)
        mask = (block_start + offs) < vocab_size
        block = tl.load(logits + row_start + block_start + offs, mask=mask)
        block = _softcap(block, softcap_thresh)
        block = block - max_val
        exp_block = tl.exp(block)
        exp_sum += tl.sum(exp_block, axis=0)
        log_sum += tl.sum(block, axis=0)

    tgt = tl.load(target + pid)
    target_val = tl.load(logits + row_start + tgt)
    target_val = _softcap(target_val, softcap_thresh)

    # compute loss with label smoothing
    smooth_factor = epsilon / vocab_size
    confidence = 1.0 - epsilon
    
    nll_loss = -(target_val - max_val - tl.log(exp_sum))
    smooth_loss = -(log_sum / vocab_size - tl.log(exp_sum))
    
    loss = confidence * nll_loss + epsilon * smooth_loss
    tl.store(output + pid, loss)

@triton.jit
def _cross_entropy_backward_kernel(
    grad_output,
    logits,
    target,
    grad_input,
    batch_size,
    vocab_size,
    stride,
    epsilon,
    softcap_thresh,
    BLOCK_SIZE: tl.constexpr,):

    pid = tl.program_id(0)
    if pid >= batch_size:
        return
    row_start = pid * stride
    max_val = float("-inf")
    exp_sum = 0.0
    smooth_factor = epsilon / vocab_size
    confidence = 1.0 - epsilon

    
    grad_scale = tl.load(grad_output + pid)
    tgt = tl.load(target + pid)

    # find max
    
    for block_start in range(0, vocab_size, BLOCK_SIZE):
        offs = tl.arange(0, BLOCK_SIZE)
        mask = (block_start + offs) < vocab_size
        block = tl.load(logits + row_start + block_start + offs, mask=mask)
        block = _softcap(block, softcap_thresh)
        block_max = tl.max(block, axis=0)
        max_val = max(max_val, block_max)

    # compute exp sum
    
    for block_start in range(0, vocab_size, BLOCK_SIZE):
        offs = tl.arange(0, BLOCK_SIZE)
        mask = (block_start + offs) < vocab_size
        block = tl.load(logits + row_start + block_start + offs, mask=mask)
        block = _softcap(block, softcap_thresh)
        exp_block = tl.exp(block - max_val)
        exp_sum += tl.sum(exp_block, axis=0)

    # Compute gradients block by block
    for block_start in range(0, vocab_size, BLOCK_SIZE):
        offs = tl.arange(0, BLOCK_SIZE)
        mask = (block_start + offs) < vocab_size
        
        block = tl.load(logits + row_start + block_start + offs, mask=mask)
        block = _softcap(block, softcap_thresh)
        
        exp_block = tl.exp(block - max_val)
        probs = exp_block / exp_sum
        
        # compute gradients with label smoothing
        is_target = (block_start + offs) == tgt
        grad_probs = confidence * probs + smooth_factor
        grad_probs = tl.where(is_target, grad_probs - confidence, grad_probs)
        
        # apply softcap gradient
        softcap_mask = block > softcap_thresh
        grad_probs = tl.where(softcap_mask, grad_probs / (1.0 + (block - softcap_thresh)), grad_probs)

        grad_probs = grad_probs * grad_scale
        tl.store(grad_input + row_start + block_start + offs, grad_probs, mask=mask)

class FastCrossEntropy(torch.autograd.Function):
    
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    def forward(ctx, logits: torch.Tensor, target: torch.Tensor, 
                epsilon: float = 0.1, softcap_thresh: float = 20.0) -> torch.Tensor:
        if not logits.is_cuda or not target.is_cuda:
            raise ValueError("Inputs must be CUDA tensors")
            
        if logits.dtype not in [torch.float16, torch.bfloat16, torch.float32]:
            raise ValueError("Logits must be float16, bfloat16 or float32")
            
        if target.dtype != torch.long:
            raise ValueError("Target must be long")

        batch_size, vocab_size = logits.shape
        device = logits.device
        
        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            batch_size_per_gpu = batch_size // world_size
            start_idx = rank * batch_size_per_gpu
            end_idx = start_idx + batch_size_per_gpu
            logits = logits[start_idx:end_idx]
            target = target[start_idx:end_idx]
            batch_size = batch_size_per_gpu

        output = torch.empty(batch_size, device=device, dtype=torch.float32)
        config = _get_optimal_config(vocab_size)
        
        grid = (batch_size,)
        _cross_entropy_fwd_kernel[grid](
            logits,
            target,
            output,
            batch_size,
            vocab_size,
            logits.stride(0),
            epsilon,
            softcap_thresh,
            config.BLOCK_SIZE,
            num_warps=config.NUM_WARPS,
            num_stages=config.NUM_STAGES
        )

        loss = output.mean()
        
        if dist.is_initialized():
            dist.all_reduce(loss)
            loss = loss / dist.get_world_size()
            
        ctx.save_for_backward(logits, target)
        ctx.config = config
        ctx.epsilon = epsilon
        ctx.softcap_thresh = softcap_thresh
        return loss

    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None, None]:
        logits, target = ctx.saved_tensors
        batch_size, vocab_size = logits.shape
        device = logits.device
        
        grad_input = torch.empty_like(logits)
        grid = (batch_size,)
        
        _cross_entropy_backward_kernel[grid](
            grad_output.expand(batch_size),
            logits,
            target,
            grad_input,
            batch_size,
            vocab_size,
            logits.stride(0),
            ctx.epsilon,
            ctx.softcap_thresh,
            ctx.config.BLOCK_SIZE,
            num_warps=ctx.config.NUM_WARPS,
            num_stages=ctx.config.NUM_STAGES
        )
        
        return grad_input, None, None, None

def cross_entropy(
    logits: torch.Tensor, 
    target: torch.Tensor,
    label_smoothing: float = 0.1,
    softcap_thresh: float = 20.0
) -> torch.Tensor:
    """
    fast cross entropy loss
    
    args:
        logits: Input tensor of shape [batch_size, vocab_size]
        target: Target indices of shape [batch_size]
        label_smoothing: Label smoothing factor (default: 0.1)
        softcap_thresh: Threshold for softcapping logits (default: 20.0)
        
    example:
        >>> logits = torch.randn(32, 50257, device='cuda')
        >>> target = torch.randint(0, 50257, (32,), device='cuda')
        >>> # Basic usage
        >>> loss = cross_entropy(logits, target)
        >>> # With custom label smoothing and softcapping
        >>> loss = cross_entropy(logits, target, label_smoothing=0.05, softcap_thresh=15.0)
        >>> loss.backward()
    """
    return FastCrossEntropy.apply(logits, target, label_smoothing, softcap_thresh)


# batch_size = 32
# vocab_size = 50257
# device = torch.device("cuda")
# 
# logits = torch.randn(batch_size, vocab_size,
                    # device=device,
                    # dtype=torch.float16,
                    # requires_grad=True)
# target = torch.randint(0, vocab_size, (batch_size,),
                    #   device=device)
# 
# loss = cross_entropy(logits, target)
# 
# loss = cross_entropy(logits, target, 
                    # label_smoothing=0.05,
                    # softcap_thresh=15.0)
# loss.backward()
