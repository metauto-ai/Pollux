from apps.Castor.model import ModelArgs


def attention_flops_per_token(n_layers, seq_len, dim, causal):
    # Formula from https://github.com/Dao-AILab/flash-attention/blob/main/benchmarks/benchmark_flash_attention.py#L27-L30
    return 3.5 * (4 * n_layers * seq_len * dim // (2 if causal else 1))


def get_num_flop_per_token(
    num_non_embed_params: int, n_layers: int, dim: int, seq_len: int
) -> int:
    return 6 * num_non_embed_params + attention_flops_per_token(
        n_layers, seq_len, dim, False
    )


def get_num_flop_per_token_in_castor_model(model_args: ModelArgs) -> int:
    """
    Calculate approximate FLOPs per token for a Castor model using the model's arguments.
    
    Args:
        model_args: The ModelArgs object containing the model configuration
        
    Returns:
        Estimated number of FLOPs per token
    """
    # Extract needed parameters from model_args
    args = model_args.diffusion_model
    n_layers = args.n_layers
    dim = args.dim
    ffn_dim_multiplier = args.ffn_dim_multiplier
    n_heads = args.n_heads
    n_kv_heads = args.n_kv_heads or n_heads
    patch_size = args.patch_size
    qk_norm = args.qk_norm
    multiple_of = args.multiple_of
    
    # Use the base_image_seq_len from scheduler as the sequence length
    seq_len = model_args.scheduler.base_image_seq_len
    
    # Calculate actual hidden dimension for FFN
    hidden_dim = int(4 * dim * ffn_dim_multiplier)
    # Make hidden_dim divisible by multiple_of
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    
    # Attention FLOPs per layer
    # QKV projections
    qkv_proj_flops = dim * dim * (2 + n_heads // n_kv_heads)
    # QK norm if enabled
    qk_norm_flops = 2 * dim if qk_norm else 0
    # Attention calculation (QK^T, softmax, attention * V)
    attn_flops = 2 * n_heads * (dim // n_heads) * seq_len + 4 * seq_len * n_heads + n_heads * (dim // n_heads) * seq_len
    # Output projection
    out_proj_flops = dim * dim
    # Total attention FLOPs
    attention_flops = qkv_proj_flops + qk_norm_flops + attn_flops + out_proj_flops
    
    # FFN FLOPs per layer
    # First projection, activation, and second projection
    ffn_flops = dim * hidden_dim + hidden_dim + hidden_dim * dim
    
    # AdaLN modulation (4 scale/gate pairs per layer)
    adaln_flops = 4 * dim
    
    # RMSNorm (2 per layer: pre-attention and pre-FFN)
    rms_norm_flops = 2 * 2 * dim  # multiply by 2 for division and multiplication operations
    
    # Rotary embeddings application
    rope_flops = 2 * dim  # rough approximation for complex multiplications
    
    # Total per layer
    per_layer_flops = attention_flops + ffn_flops + adaln_flops + rms_norm_flops + rope_flops
    
    # Total across all layers
    total_flops = n_layers * per_layer_flops
    
    # Image embedding/unembedding operations
    patch_embedding_flops = (patch_size * patch_size * dim) // seq_len
    
    return total_flops + patch_embedding_flops


def calculate_total_flops(model_args, batch_size, input_seq_len=None):
    # Get per-token FLOPs
    flops_per_token = get_num_flop_per_token_in_castor_model(model_args)
    
    # Use provided input_seq_len or fall back to model's base_image_seq_len
    seq_len = input_seq_len or model_args.scheduler.base_image_seq_len
    
    # Calculate total FLOPs for the batch
    total_flops = flops_per_token * seq_len * batch_size
    
    return {
        "flops_per_token": flops_per_token,
        "total_flops": total_flops,
        "total_tflops": total_flops / 1e12
    }


if __name__ == "__main__":
    from apps.Castor.model import ModelArgs
    from apps.Castor.modules.transformer import TransformerArgs
    from apps.Castor.modules.schedulers import SchedulerArgs
    from apps.main.utils.cal_flops import get_num_flop_per_token_in_castor_model
    
    # Create a test model config
    diffusion_model = TransformerArgs(
        dim=2048,
        n_layers=24,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.5,
        multiple_of=256,
        patch_size=2,
        qk_norm=False,
        liger_rms_norm=True,
        liger_ffn=True,
        liger_rotary_emb=False,
        shared_adaLN=True
    )
    
    scheduler = SchedulerArgs(base_image_seq_len=256)
    
    model_args = ModelArgs(
        diffusion_model=diffusion_model,
        scheduler=scheduler
    )

    stats = calculate_total_flops(model_args, 32, 512)
    print(stats)