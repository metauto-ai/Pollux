import torch

def pad_flat_tokens_to_multiple(
    x_patched, freqs_cis, seqlens, cu_seqlens, max_seqlen, modulation_signal,
    multiple:    int = 8,
):
    """

    Returns: x_patched, freqs_cis, seqlens, cu_seqlens, max_seqlen, modulation_signal, pad_len
    """
   
    assert x_patched.dim() == 2, "x_patched must be 2D"
    N, D = x_patched.shape
    mod = N % multiple
    if mod == 0:
        return x_patched, freqs_cis, seqlens, cu_seqlens, max_seqlen, modulation_signal, 0

    pad = multiple - mod

    # 1. dummy token vectors
    dummy_x = torch.zeros((pad, D), dtype=x_patched.dtype, device=x_patched.device)
    x_patched = torch.cat([x_patched, dummy_x], dim=0)  # (N+pad, D)

    # 2. dummy freq_cis vectors
    # freqs_cis can have shape (N, ..., F_rope)
    dummy_freq_shape = (pad,) + freqs_cis.shape[1:]
    dummy_freq = torch.zeros(dummy_freq_shape, dtype=freqs_cis.dtype, device=freqs_cis.device)
    freqs_cis = torch.cat([freqs_cis, dummy_freq], dim=0)  # (N+pad, *other_dims, F_rope)

    # 3. seqlens: Add a new sequence of length 'pad'
    # Original seqlens shape: (B,)
    seqlens = torch.cat([seqlens, torch.tensor([pad], dtype=seqlens.dtype, device=seqlens.device)], dim=0) # Shape: (B+1,)

    # 4. cu_seqlens: Add the new sequence length to the cumulative sum
    # Original cu_seqlens shape: (B+1,)
    # cu_seqlens[-1] is total tokens N. New last element is N + pad.
    cu_seqlens = torch.cat([cu_seqlens, (cu_seqlens[-1] + seqlens[-1]).unsqueeze(0)], dim=0) # Shape: (B+2,)

    # 5. max_seqlen: Update with the length of the new dummy sequence
    max_seqlen = max(max_seqlen, seqlens[-1].item()) # seqlens[-1] is 'pad'

    # 6. modulation_signal: Add a dummy modulation signal for the new sequence
    # Original modulation_signal shape: (B, mod_dim)
    # Ensure the dummy modulation signal has the correct feature dimension
    mod_dim = modulation_signal.shape[1:]
    dummy_mod_shape = (1,) + mod_dim

    dummy_modulation = torch.zeros(dummy_mod_shape, dtype=modulation_signal.dtype, device=modulation_signal.device)
    modulation_signal = torch.cat([modulation_signal, dummy_modulation], dim=0) # Shape: (B+1, mod_dim)


    return x_patched, freqs_cis, seqlens, cu_seqlens, max_seqlen, modulation_signal, pad


def unpad_flat_tokens(
    x:  torch.Tensor,
    padded_len: int,
):
    """
    Remove the last `padded_len` rows from x
    If padded_len == 0, returns the tensors unchanged.
    """
    if x is None:
        return None
    assert x.dim() ==2, f"should have (Total tokens, dim) shape got {x.shape}"

    if padded_len == 0:
        return x

    # slice off the dummy rows
    x_unpadded    = x[:-padded_len]
    return x_unpadded
