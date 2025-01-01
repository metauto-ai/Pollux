import triton
import triton.language as tl
import torch
import torch.nn as nn
from typing import Tuple
import math
from configure import calculate_settings

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
    
    rms = tl.sqrt(sum_sqr_x / inp_row_stride).to(tl.float16)
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
        
        if scale is None:
            scale = torch.ones(cols, dtype=torch.float16, device='cuda')
        
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
