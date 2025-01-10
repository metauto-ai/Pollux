import torch
import triton
import triton.language as tl

@triton.jit
def silu_fwd_kernel(
    X, OUT,
    stride_xm, stride_xn,
    stride_om, stride_on,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    
    pid_m = pid // num_pid_m
    pid_n = pid % num_pid_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    x_ptrs = X + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    
    out = x * tl.sigmoid(x)
    
    out_ptrs = OUT + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, out, mask=mask)

@triton.jit
def silu_bwd_kernel(
    X, GRAD_OUTPUT, GRAD_X,
    stride_xm, stride_xn,
    stride_gm, stride_gn,
    stride_dxm, stride_dxn,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    
    pid_m = pid // num_pid_m
    pid_n = pid % num_pid_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    x_ptrs = X + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    grad_ptrs = GRAD_OUTPUT + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn
    
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    grad_output = tl.load(grad_ptrs, mask=mask, other=0.0)
    
    # grad_output * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))
    sig_x = tl.sigmoid(x)
    grad_x = grad_output * (sig_x + x * sig_x * (1 - sig_x))
    
    grad_x_ptrs = GRAD_X + offs_m[:, None] * stride_dxm + offs_n[None, :] * stride_dxn
    tl.store(grad_x_ptrs, grad_x, mask=mask)

class SiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if not x.is_contiguous():
            x = x.contiguous()
        
        M, N = x.shape
        out = torch.empty_like(x)
        
        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
        
        silu_fwd_kernel[grid](
            x, out,
            x.stride(0), x.stride(1),
            out.stride(0), out.stride(1),
            M, N,
            BLOCK_M=32,
            BLOCK_N=32
        )
        
        ctx.save_for_backward(x)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
            
        M, N = x.shape
        grad_x = torch.empty_like(x)
        
        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
        
        silu_bwd_kernel[grid](
            x, grad_output, grad_x,
            x.stride(0), x.stride(1),
            grad_output.stride(0), grad_output.stride(1),
            grad_x.stride(0), grad_x.stride(1),
            M, N,
            BLOCK_M=32,
            BLOCK_N=32
        )
        
        return grad_x

def silu(x):
    return SiLU.apply(x)
