import torch
import torch.utils.benchmark as benchmark
from torch._inductor import config

model = torch.nn.Sequential(
    *[torch.nn.Linear(1024, 1024, False, device="cuda") for _ in range(10)]
)


input = torch.rand(1024, device="cuda")
output = model(input)
output.sum().backward()

use_scheduler = False
cudagraphs = True
cppwrapper = True

if cudagraphs:
    config.triton.cudagraphs = True
if cppwrapper:
    config.cpp_wrapper = True

for p in model.parameters():
    p.grad = torch.rand_like(p)
    torch._dynamo.mark_static_address(p.grad)

opt = torch.optim.Adam(
    model.parameters(),
    lr=torch.tensor(0.01, device="cuda"),
    foreach=True,
    capturable=True,
    fused=False,
)
opt_fused = torch.optim.Adam(
    model.parameters(),
    lr=torch.tensor(0.01),
    foreach=False,
    capturable=True,
    fused=True,
)

if use_scheduler:
    scheduler = torch.optim.lr_scheduler.LinearLR(opt)

    @torch.compile(fullgraph=False)
    def fn():
        opt.step()
        scheduler.step()
else:
    @torch.compile(fullgraph=False)
    def fn():
        opt.step()


def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    return t0.blocked_autorange().mean * 1e6


# Warmup runs to compile the function
for _ in range(5):
    fn()

benchmark_torch_function_in_microseconds(opt.step)
benchmark_torch_function_in_microseconds(fn)
benchmark_torch_function_in_microseconds(opt_fused.step)

eager_runtime = benchmark_torch_function_in_microseconds(opt.step)
compiled_runtime = benchmark_torch_function_in_microseconds(fn)
fused_runtime = benchmark_torch_function_in_microseconds(opt_fused.step)

print(f"eager runtime: {eager_runtime}us")
print(f"compiled runtime {cudagraphs=} {cppwrapper=}: {compiled_runtime}us")
print(f"fused runtime: {fused_runtime}us")