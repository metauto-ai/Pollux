import torch
import triton
import triton.language as tl

@triton.jit
def silu_fwd_kernel(
    X, OUT,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(X + offsets, mask=mask)
    out = x * tl.sigmoid(x)
    tl.store(OUT + offsets, out, mask=mask)

@triton.jit
def silu_bwd_kernel(
    X, GRAD_OUTPUT, GRAD_X,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(X + offsets, mask=mask)
    grad_output = tl.load(GRAD_OUTPUT + offsets, mask=mask)
    
    sig_x = tl.sigmoid(x)
    grad_x = grad_output * (sig_x + x * sig_x * (1 - sig_x))
    tl.store(GRAD_X + offsets, grad_x, mask=mask)

class SiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if not x.is_contiguous():
            x = x.contiguous()
        
        n_elements = x.numel()
        out = torch.empty_like(x)
        
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        
        silu_fwd_kernel[grid](
            x, out,
            n_elements,
            BLOCK_SIZE=1024,
        )
        
        ctx.save_for_backward(x)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
            
        n_elements = x.numel()
        grad_x = torch.empty_like(x)
        
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        
        silu_bwd_kernel[grid](
            x, grad_output, grad_x,
            n_elements,
            BLOCK_SIZE=1024,
        )
        
        return grad_x

def silu(x):
    return SiLU.apply(x)
