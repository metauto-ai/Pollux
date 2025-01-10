import triton
import triton.language as tl
import torch
import torch.nn as nn
from typing import Tuple
import math
def calculate_flops(batch_size: int, seq_len: int, hidden_dim: int) -> int:
    flops_per_seq = (
        hidden_dim +          # Square each element
        (hidden_dim - 1) +    # Sum reduction
        1 +                   # Division by hidden_dim
        1 +                   # Square root
        (2 * hidden_dim)      # Final division and scale multiplication
    )
    return flops_per_seq * batch_size * seq_len

@triton.jit
def _rmsn_fwd_kernel(
    inp_ptr, 
    out_ptr, 
    scale_ptr,
    inp_row_stride,
    out_row_stride,
    g_row_stride,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    offset = pid * inp_row_stride + cols
    mask = cols < inp_row_stride
    
    x = tl.load(inp_ptr + offset, mask=mask).to(tl.float16)
    
    x_f32 = x.to(tl.float32)
    sum_sqr_x = tl.sum(x_f32 * x_f32, axis=0) 
    
    rms = tl.sqrt(sum_sqr_x / inp_row_stride + 1e-6).to(tl.float16)
    scale = tl.load(scale_ptr + cols, mask=mask).to(tl.float16)
    
    out = (x / rms) * scale
    
    tl.store(out_ptr + offset, out, mask=mask)

@triton.jit
def _rmsn_bwd_kernel(
    grad_out_ptr,
    inp_ptr,
    scale_ptr,
    grad_input_ptr,
    grad_scale_ptr,
    inp_row_stride,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    offset = pid * inp_row_stride + cols
    mask = cols < inp_row_stride
    
    grad_out = tl.load(grad_out_ptr + offset, mask=mask).to(tl.float32)
    x = tl.load(inp_ptr + offset, mask=mask).to(tl.float32)
    scale = tl.load(scale_ptr + cols, mask=mask).to(tl.float32)
    
    variance = tl.sum(x * x, axis=0) / inp_row_stride
    inv_std = 1.0 / tl.sqrt(variance)
    x_norm = x * inv_std
    
    dx_norm = grad_out * scale
    
    # Compute variance gradient
    # First term: dx_norm * x * -1/(2 * variance^(3/2))
    variance_grad = dx_norm * x * (-0.5 / (variance * tl.sqrt(variance)))
    
    # Sum for the second term of input gradient
    sum_dx_norm_x = tl.sum(dx_norm * x_norm, axis=0)
    
    # Compute final input gradient
    # dx = (1/sqrt(variance)) * (dx_norm - mean(dx_norm * x_norm) * x_norm)
    grad_input = inv_std * (dx_norm - (sum_dx_norm_x / inp_row_stride) * x_norm)
    
    tl.store(grad_input_ptr + offset, grad_input.to(tl.float16), mask=mask)
    
    if pid == 0:
        grad_scale = tl.sum(grad_out * x_norm, axis=0).to(tl.float16)
        tl.atomic_add(grad_scale_ptr + cols, grad_scale, mask=mask)

class RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale=None):
        dims = x.shape
        cols = dims[-1]
        x = x.view(-1, cols)
        
        # Create or use provided scale
        if scale is None:
            scale = torch.ones(cols, dtype=torch.float16, device='cuda')
        
        # Initialize output
        out = torch.zeros_like(x, dtype=torch.float16)
        
        block_size, num_warps = calculate_settings(cols)
        rows = x.shape[0]
        
        _rmsn_fwd_kernel[(rows,)](
            x,
            out,
            scale,
            cols,
            cols,
            cols,
            BLOCK_SIZE=block_size,
            num_warps=num_warps
        )
        
        ctx.save_for_backward(x, scale)
        ctx.dims = dims
        ctx.cols = cols
        ctx.rows = rows
        
        return out.view(dims)

    @staticmethod
    def backward(ctx, grad_output):
        x, scale = ctx.saved_tensors
        dims = ctx.dims
        cols = ctx.cols
        rows = ctx.rows
        
        grad_output = grad_output.contiguous()
        if grad_output.dtype != torch.float16:
            grad_output = grad_output.half()
        grad_output = grad_output.view(-1, cols)
        
        grad_input = torch.zeros_like(x)
        grad_scale = torch.zeros_like(scale)
        
        block_size, num_warps = calculate_settings(cols)
        
        _rmsn_bwd_kernel[(rows,)](
            grad_output,
            x,
            scale,
            grad_input,
            grad_scale,
            cols,
            BLOCK_SIZE=block_size,
            num_warps=num_warps
        )
        
        return grad_input.view(dims), grad_scale if scale.requires_grad else None

def rmsnorm(x: torch.Tensor):
    if x.dtype != torch.float16:
        x = x.half()
    
    scale = torch.ones(x.shape[-1], dtype=torch.float16, device='cuda')
    
    rms = torch.sqrt((x * x).mean(dim=-1, keepdim=True)).half()
    return (x / rms) * scale
