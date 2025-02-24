import triton
import triton.language as tl
import torch
import torch.nn as nn
from typing import Tuple

MAX_FUSED_SIZE: int = 65536
EPSILON: float = 1e-5  # Match PyTorch's default LayerNorm epsilon
MAX_GRAD_NORM: float = 1.0  # Gradient clipping threshold

def calculate_flops(batch_size: int, seq_len: int, hidden_dim: int) -> int:
    return 5 * batch_size * seq_len * hidden_dim  # Approximation for RMSNorm

def calculate_settings(n: int) -> Tuple[int, int]:
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(f"Dimension {n} exceeds maximum block size {MAX_FUSED_SIZE}")
    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps

@triton.jit
def rmsn_fwd_kernel(
    inp_ptr, out_ptr, scale_ptr,
    inp_row_stride, out_row_stride, 
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    offset = row_idx * inp_row_stride + cols

    x = tl.load(inp_ptr + offset, mask=mask)
    x_f32 = x.to(tl.float32)  

    sum_sqr = tl.sum(x_f32 * x_f32, axis=0)
    variance = (sum_sqr + eps) / n_cols  
    rms = tl.sqrt(variance)
    rms_safe = tl.where(rms < eps, eps, rms)  
    rms_bf16 = rms_safe.to(tl.bfloat16)

    scale = tl.load(scale_ptr + cols, mask=mask)
    output = (x / rms_bf16) * scale
    tl.store(out_ptr + offset, output, mask=mask)

@triton.jit
def rmsn_bwd_kernel(
    grad_out_ptr, inp_ptr, scale_ptr,
    grad_input_ptr, grad_scale_ptr,
    n_cols, eps, max_grad_norm,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    offset = row_idx * n_cols + cols

    grad_out = tl.load(grad_out_ptr + offset, mask=mask)
    grad_out = tl.where(tl.abs(grad_out) > max_grad_norm,
                       tl.where(grad_out > 0, max_grad_norm, -max_grad_norm), 
                       grad_out)
    
    x = tl.load(inp_ptr + offset, mask=mask)
    scale = tl.load(scale_ptr + cols, mask=mask)

    x_f32 = x.to(tl.float32)
    sum_sqr = tl.sum(x_f32 * x_f32, axis=0)
    variance = (sum_sqr + eps) / n_cols
    variance_clamped = tl.clamp(variance, 1e-6, 1e6)   
    inv_rms = 1.0 / tl.sqrt(variance_clamped)
    inv_rms_bf16 = inv_rms.to(tl.bfloat16)

    x_norm = x * inv_rms_bf16
    dx_norm = grad_out * scale

    dx_norm_f32 = dx_norm.to(tl.float32)
    x_norm_f32 = x_norm.to(tl.float32)
    x_f32 = x.to(tl.float32)

    dot_product = tl.sum(dx_norm_f32 * x_norm_f32, axis=0)
    grad_input = inv_rms_bf16 * (dx_norm - (x_norm * (dot_product / n_cols).to(tl.bfloat16)))
    tl.store(grad_input_ptr + offset, grad_input, mask=mask)

    if row_idx == 0:
        grad_scale = tl.sum(grad_out.to(tl.float32) * x_norm_f32, axis=0)
        tl.atomic_add(grad_scale_ptr + cols, grad_scale, mask=mask)

class RMS_Norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        shape = x.shape
        n_cols = shape[-1]
        x_2d = x.reshape(-1, n_cols)
        ctx.shape = shape
        
        out = torch.empty_like(x_2d)
        
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)
        n_rows = x_2d.shape[0]
        
        rmsn_fwd_kernel[(n_rows,)](
            x_2d, out, scale,
            x_2d.stride(0), out.stride(0),
            n_cols, EPSILON,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps
        )
        
        ctx.save_for_backward(x_2d, scale)
        ctx.n_cols = n_cols
        return out.view(shape)

    @staticmethod
    def backward(ctx, grad_output):
        x_2d, scale = ctx.saved_tensors
        n_cols = ctx.n_cols
        
        grad_input = torch.empty_like(x_2d)
        grad_scale = torch.zeros_like(scale, dtype=torch.float32)
        
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)
        n_rows = x_2d.shape[0]
        
        rmsn_bwd_kernel[(n_rows,)](
            grad_output.contiguous().view_as(x_2d),
            x_2d,
            scale,
            grad_input,
            grad_scale,
            n_cols, EPSILON, MAX_GRAD_NORM, 
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps
        )
        
        # Cast accumulated gradients to bf16
        return grad_input.view(ctx.shape), grad_scale.to(torch.bfloat16)

def test_backward():
    x = torch.randn(32, 128, 512, dtype=torch.float16, device='cuda', requires_grad=True)
    
    out = RMS_Norm.apply(x)
    
    grad_output = torch.randn_like(out)
    
    out.backward(grad_output)
    
    print("Gradient shape:", x.grad.shape)
    print("Gradient contains NaN:", torch.isnan(x.grad).any())
    print("Gradient contains Inf:", torch.isinf(x.grad).any())
def rmsnorm(x: torch.Tensor):
    if x.dtype != torch.float16:
        x = x.half()
    
    scale = torch.ones(x.shape[-1], dtype=torch.float16, device='cuda')
    
    rms = torch.sqrt((x * x).mean(dim=-1, keepdim=True)).half()
    return (x / rms) * scale

def get_gpu_memory_info():
    t = torch.cuda.get_device_properties(0)
    memory_used = torch.cuda.memory_allocated() / (1024**2)  
    memory_total = t.total_memory / (1024**2) 
    return memory_used, memory_total

def get_gpu_info():
    t = torch.cuda.get_device_properties(0)
    return {
        'name': t.name,
        'compute_capability': f"{t.major}.{t.minor}",
    }

def benchmark():
    batch_size = 512
    seq_len = 1024
    hidden_dim = 512
    num_iterations = 100
    warmup_iterations = 10
    
    x = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16, device='cuda')
    
    total_flops = calculate_flops(batch_size, seq_len, hidden_dim)
    
    gpu_info = get_gpu_info()
    print("\nGPU Information:")
    for key, value in gpu_info.items():
        print(f"{key}: {value}")
    
    torch.cuda.synchronize()
    for _ in range(warmup_iterations):
        torch_rms = rmsnorm(x)
        triton_rms = RMS_Norm.apply(x)
    torch.cuda.synchronize()
    
    import time
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iterations):
        torch_rms = rmsnorm(x)
    torch.cuda.synchronize()
    torch_time = (time.perf_counter() - start) / num_iterations
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iterations):
        triton_rms = RMS_Norm.apply(x)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / num_iterations
    
    torch_flops = total_flops / torch_time
    triton_flops = total_flops / triton_time
    
    memory_used, memory_total = get_gpu_memory_info()
    
    print("\nPerformance Metrics:")
    print(f"Input shape: {x.shape}")
    print(f"FLOPs per iteration: {total_flops:,}")
    print(f"\nPyTorch implementation:")
    print(f"Time: {torch_time*1000:.3f} ms")
    print(f"FLOP/s: {torch_flops/1e9:.2f} GFLOP/s")
    print(f"\nTriton implementation:")
    print(f"Time: {triton_time*1000:.3f} ms")
    print(f"FLOP/s: {triton_flops/1e9:.2f} GFLOP/s")
    print(f"Speedup: {torch_time/triton_time:.2f}x")
    
    print("\nMemory Usage:")
    print(f"Used: {memory_used:.2f} MB")
    print(f"Total: {memory_total:.2f} MB")
    print(f"Utilization: {(memory_used/memory_total)*100:.2f}%")
    
    print(f"Implementations match: {torch.testing.assert_close(torch_rms, triton_rms)}")


class TorchRMSNorm(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(hidden_size, dtype=torch.float16))
        
    def forward(self, x):
        x_f32 = x.float()
        variance = x_f32.pow(2).mean(-1, keepdim=True)
        x_norm = x_f32 / variance.sqrt()
        
        return (x_norm * self.scale)

def compare_gradients(batch_size=32, seq_len=128, hidden_size=512, atol=1e-2, rtol=1e-2):

    print("\nGradient Verification Test")
    print("-" * 50)
    
    x = torch.randn(batch_size, seq_len, hidden_size, 
                    dtype=torch.float16, 
                    device='cuda', 
                    requires_grad=True)
    grad_output = torch.randn_like(x)
    
    x_torch = x.clone().detach().requires_grad_(True)
    
    triton_out = RMS_Norm.apply(x)
    triton_out.backward(grad_output)
    triton_grad = x.grad.clone()
    
    torch_model = TorchRMSNorm(hidden_size).cuda()
    torch_out = torch_model(x_torch)
    torch_out.backward(grad_output)
    torch_grad = x_torch.grad.clone()
    
    output_match = torch.allclose(triton_out, torch_out, atol=atol, rtol=rtol)
    grad_match = torch.allclose(triton_grad, torch_grad, atol=atol, rtol=rtol)
    
    print(f"Test Configuration:")
    print(f"- Batch Size: {batch_size}")
    print(f"- Sequence Length: {seq_len}")
    print(f"- Hidden Size: {hidden_size}")
    print(f"- Tolerance: atol={atol}, rtol={rtol}")
    print("\nResults:")
    print(f"- Forward outputs match: {output_match}")
    print(f"- Backward gradients match: {grad_match}")
    
    if not output_match or not grad_match:
        print("\nDetailed Error Analysis:")
        forward_diff = (triton_out - torch_out).abs()
        print(f"Forward max difference: {forward_diff.max().item():.6f}")
        print(f"Forward mean difference: {forward_diff.mean().item():.6f}")
        
        backward_diff = (triton_grad - torch_grad).abs()
        print(f"Backward max difference: {backward_diff.max().item():.6f}")
        print(f"Backward mean difference: {backward_diff.mean().item():.6f}")

    
    return output_match and grad_match

def run_gradient_tests():

    print("Running Gradient Test Suite")
    print("=" * 50)
    
    test_configs = [
        (32, 128, 512),    
        (1, 128, 512),    
        (32, 1, 512),      
        (32, 128, 128),    
        (64, 256, 1024),   
    ]
    
    all_passed = True
    for batch, seq, hidden in test_configs:
        print(f"\nTesting configuration: batch={batch}, seq_len={seq}, hidden={hidden}")
        passed = compare_gradients(batch, seq, hidden)
        all_passed &= passed
        
    print("\nFinal Results:")
    print(f"All tests {'passed' if all_passed else 'failed'}")
    
    return all_passed

if __name__ == "__main__":
    run_gradient_tests()

# if __name__ == "__main__":
#     benchmark()
# Running Gradient Test Suite
# ==================================================

# Testing configuration: batch=32, seq_len=128, hidden=512

# Gradient Verification Test
# --------------------------------------------------
# Test Configuration:
# - Batch Size: 32
# - Sequence Length: 128
# - Hidden Size: 512
# - Tolerance: atol=0.01, rtol=0.01

# Results:
# - Forward outputs match: True
# - Backward gradients match: True

# Testing configuration: batch=1, seq_len=128, hidden=512

# Gradient Verification Test
# --------------------------------------------------
# Test Configuration:
# - Batch Size: 1
# - Sequence Length: 128
# - Hidden Size: 512
# - Tolerance: atol=0.01, rtol=0.01

# Results:
# - Forward outputs match: True
# - Backward gradients match: True

# Testing configuration: batch=32, seq_len=1, hidden=512

# Gradient Verification Test
# --------------------------------------------------
# Test Configuration:
# - Batch Size: 32
# - Sequence Length: 1
# - Hidden Size: 512
# - Tolerance: atol=0.01, rtol=0.01

# Results:
# - Forward outputs match: True
# - Backward gradients match: True

# Testing configuration: batch=32, seq_len=128, hidden=128

# Gradient Verification Test
# --------------------------------------------------
# Test Configuration:
# - Batch Size: 32
# - Sequence Length: 128
# - Hidden Size: 128
# - Tolerance: atol=0.01, rtol=0.01

# Results:
# - Forward outputs match: True
# - Backward gradients match: True

# Testing configuration: batch=64, seq_len=256, hidden=1024

# Gradient Verification Test
# --------------------------------------------------
# Test Configuration:
# - Batch Size: 64
# - Sequence Length: 256
# - Hidden Size: 1024
# - Tolerance: atol=0.01, rtol=0.01

# Results:
# - Forward outputs match: True
# - Backward gradients match: True

# Final Results:
# All tests passed