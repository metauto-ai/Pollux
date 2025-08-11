from apps.Castor.model import Castor
from apps.Castor.modules.transformer import TransformerArgs
from lingua.metrics import get_num_params


def attention_flops_per_token(batch, seqlen, headdim, nheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)


def get_num_flop_per_token(params: int, mode="fwd") -> int:
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    if mode == "fwd":
        return 2 * params
    elif mode == "bwd":
        return 4 * params
    else:
        return 6 * params


def estimate_vae_flops(
    n_params: int,
    B: int,
    H_in: int,
    W_in: int,
    mode: str = "fwd",
    k_arch_approx: float = 0.01,
) -> float:
    """
    Estimates FLOPs for a convolutional encoder part of a model.

    This function provides an approximation based on the total number of
    parameters in the convolutional encoder, input dimensions, and an
    architectural coefficient (k_arch_approx).

    The base for FLOPs calculation comes from:
    FLOPs_fwd_per_sample_approx = 2 * n_params * (k_arch_approx * H_in * W_in)
    This means for each parameter, it's involved in (k_arch_approx * H_in * W_in)
    effective spatial multiply-accumulate operations on average, and each MAC is 2 FLOPs.

    Args:
        n_params: Total number of parameters in the convolutional layers of the encoder.
        batch_size: Input batch size.
        H_in: Input image height.
        W_in: Input image width.
        mode: Calculation mode.
              - "fwd": Forward pass FLOPs (approx. 2 * N_eff).
              - "bwd": Backward pass FLOPs (approx. 4 * N_eff, assumes backward is 2x fwd).
              - "fwd_bwd": Total training FLOPs (approx. 6 * N_eff, for fwd + bwd).
              (where N_eff = n_params * k_arch_approx * H_in * W_in for a single sample)
        k_arch_approx: Architectural coefficient. This dimensionless factor represents
                       the parameter-weighted average output feature map area
                       normalized by the input feature map area. It's calculated as:
                       k_arch = (Forward FLOPs for one sample) / (2 * n_params * H_in * W_in)
                       A typical value for ResNet-like architectures is around 0.0016.
                       For a specific model, you can calculate its k_arch if you know
                       its forward FLOPs for a given input size and its parameter count.

    Returns:
        Estimated FLOPs for the given mode and batch size.
    """
    assert mode in ["fwd", "bwd", "fwd_bwd"]

    if mode == "fwd":
        # Forward FLOPs = (2 * k_arch_approx) * base_op_count
        total_flops = (2.0 * k_arch_approx) * n_params
    elif mode == "fwd_bwd":
        # Total training FLOPs = (6 * k_arch_approx) * base_op_count
        total_flops = (6.0 * k_arch_approx) * n_params
    else:
        # Backward FLOPs = (4 * k_arch_approx) * base_op_count
        total_flops = (4.0 * k_arch_approx) * n_params

    return total_flops * B * H_in * W_in


def estimate_mfu(self, fwdbwd_per_iter, dt):
    """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
    # first estimate the number of flops we do per iteration.
    # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
    N = self.get_num_params()
    cfg = self.config
    L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
    flops_per_token = 6 * N + 12 * L * H * Q * T
    flops_per_fwdbwd = flops_per_token * T
    flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
    # express our flops throughput as ratio of A100 bfloat16 peak flops
    flops_achieved = flops_per_iter * (1.0 / dt)  # per second
    flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
    mfu = flops_achieved / flops_promised
    return mfu


theoretical_flops = {
    "a100_bf16": 312e12,
    "h100_bf16": 989e12,
}


class FlopsMeter:
    def __init__(
        self, args: TransformerArgs, model: Castor, device="h100", dtype="bf16"
    ):
        self.diffusion_params = get_num_params(model.diffusion_transformer)
        self.diffusion_num_layers = args.diffusion_model.n_layers
        self.diffusion_num_heads = args.diffusion_model.n_heads
        self.diffusion_headdim = (
            args.diffusion_model.dim // args.diffusion_model.n_heads
        )
        self.diffusion_dim = args.diffusion_model.dim

        self.cond_params = get_num_params(model.text_encoder.model)
        self.cond_dim = model.text_encoder.dim()
        self.cond_num_layers = len(model.text_encoder.model.language_model.layers)
        self.cond_num_heads = model.text_encoder.model.language_model.config.num_attention_heads
        self.cond_headdim = self.cond_dim // self.cond_num_heads

        self.vision_params = get_num_params(model.vision_encoder.model)
        self.vision_num_layers = model.vision_encoder.model.config.num_hidden_layers
        self.vision_num_heads = model.vision_encoder.model.config.num_attention_heads
        self.vision_dim = model.vision_encoder.model.config.hidden_size
        self.vision_patch_size = model.vision_encoder.model.config.patch_size
        self.vision_headdim = self.vision_dim // self.vision_num_heads

        self.vae_params = get_num_params(model.compressor.vae)

        self.diffusion_flops = 0
        self.vision_encoder_flops = 0
        self.vae_flops = 0
        self.text_encoder_flops = 0

        self.theoretical_flops = theoretical_flops[f"{device}_{dtype}"]

    def log_diffusion_flops(self, input_shape):
        non_atten_flops = (
            get_num_flop_per_token(self.diffusion_params, mode="fwd_bwd")
            * input_shape[0]
            * input_shape[1]
        )

        attention_flops = (
            attention_flops_per_token(
                batch=input_shape[0],
                seqlen=input_shape[1],
                headdim=self.diffusion_headdim,
                nheads=self.diffusion_num_heads,
                causal=False,
                mode="fwd_bwd",
            )
            * self.diffusion_num_layers
        )
        self.diffusion_flops += non_atten_flops + attention_flops

    def log_vision_encoder_flops(self, input_shape):
        B, C, H, W = input_shape
        non_atten_flops = (
            get_num_flop_per_token(self.vision_params, mode="fwd")
            * B
            * (H // self.vision_patch_size)
            * (W // self.vision_patch_size)
        )

        attention_flops = (
            attention_flops_per_token(
                batch=B,
                seqlen=(H // self.vision_patch_size) * (W // self.vision_patch_size),
                headdim=self.vision_headdim,
                nheads=self.vision_num_heads,
                causal=False,
                mode="fwd",
            )
            * self.vision_num_layers
        )
        self.vision_encoder_flops += non_atten_flops + attention_flops

    def log_vae_flops(self, input_shape):
        B, C, H, W = input_shape
        self.vae_flops += estimate_vae_flops(self.vae_params, B, H, W, mode="fwd")

    def log_text_encoder_flops(self, input_shape):
        non_atten_flops = (
            get_num_flop_per_token(self.cond_params, mode="fwd")
            * input_shape[0]
            * input_shape[1]
        )

        attention_flops = (
            attention_flops_per_token(
                batch=input_shape[0],
                seqlen=input_shape[1],
                headdim=self.cond_headdim,
                nheads=self.cond_num_heads,
                causal=False,
                mode="fwd",
            )
            * self.cond_num_layers
        )
        self.text_encoder_flops += non_atten_flops + attention_flops

    def get_mfu(self, time_delta):
        return (
            self.diffusion_flops
            + self.vision_encoder_flops
            + self.vae_flops
            + self.text_encoder_flops
        ) / (self.theoretical_flops * time_delta)

    def get_total_flops(self, time_delta):
        return (
            self.diffusion_flops
            + self.vision_encoder_flops
            + self.vae_flops
            + self.text_encoder_flops
        ) / time_delta

    def reset(self):
        self.diffusion_flops = 0
        self.vision_encoder_flops = 0
        self.vae_flops = 0
        self.text_encoder_flops = 0
