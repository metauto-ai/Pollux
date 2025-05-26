import math
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Union

import torch
from flash_attn.bert_padding import (index_first_axis, pad_input,  # noqa
                                     unpad_input)
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import (BlockMask, _mask_mod_signature,
                                               create_block_mask,
                                               flex_attention)
from xformers.ops import AttentionBias, fmha
from liger_kernel.transformers import LigerSwiGLUMLP, LigerRMSNorm, liger_rotary_pos_emb
from types import SimpleNamespace
from mup import MuReadout

# fa3 
from flash_attn_interface import flash_attn_varlen_func


flex_attention_comp = torch.compile(flex_attention)


def layer_init_kaiming_normal(x):
    nn.init.kaiming_normal_(x.weight, a=1, mode='fan_in')
    if x.bias is not None:
        nn.init.constant_(x.bias, 0.)


@dataclass
class BaseTransformerArgs:
    dim: int = 512
    n_layers: int = 8
    align_layer: int = 8
    head_dim: Optional[int] = None
    n_heads: Optional[int] = None
    n_kv_heads: Optional[int] = None

    ffn_dim_multiplier: Optional[float] = None

    multiple_of: int = 256

    norm_eps: float = 1e-5

    rope_theta: float = 10000.0

    init_base_std: Optional[float] = None
    init_std_factor: str = "disabled"

    max_seqlen: int = 1024

    qk_norm: bool = True
    liger_ffn: bool = True
    liger_rms_norm: bool = True
    liger_rotary_emb: bool = False


def nearest_multiple_of_8(x):
    return ((x + 7) // 8) * 8


def cross_entropy(pred, target, **kwargs):
    return F.nll_loss(
        F.log_softmax(pred.flatten(end_dim=-2).float(), -1),
        target.flatten(end_dim=-1),
        **kwargs,
    )


def repeat_kv(x: torch.Tensor, n_rep: int, dim: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    assert dim == 2, "Only dim=2 is supported. Check the implementation for other dims."
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()

    cos, sin = freqs.cos(), freqs.sin()

    return torch.stack((cos, -sin, sin, cos), dim=-1).view(*freqs.size(), 2, 2)


def precompute_2d_freqs_cls(
    dim: int,
    end: int,
    theta: float = 10000.0,
):
    """
    Precompute the frequency tensor for complex exponentials (cis) with
    given dimensions.

    This function calculates a frequency tensor with complex exponentials
    using the given dimension 'dim' and the end index 'end'. The 'theta'
    parameter scales the frequencies. The returned tensor contains complex
    values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation.
            Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex
            exponentials.
    """

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()

    cos, sin = freqs.cos(), freqs.sin()

    freqs_cis = torch.stack((cos, -sin, sin, cos), dim=-1).view(*freqs.size(), 2, 2)

    freqs_cis_h = freqs_cis.view(end, 1, dim // 4, 2, 2).repeat(1, end, 1, 1, 1)
    freqs_cis_w = freqs_cis.view(1, end, dim // 4, 2, 2).repeat(end, 1, 1, 1, 1)
    freqs_cis = torch.cat([freqs_cis_h, freqs_cis_w], dim=2)

    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    seq_dim: int,
    freqs_cis: torch.Tensor,
    liger_rotary_emb: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if liger_rotary_emb:
        sin = freqs_cis[:, :, 1, 0]
        cos = freqs_cis[:, :, 0, 0]
        xq, xk = liger_rotary_pos_emb(xq, xk, cos, sin)
        return xq, xk
    if xq.ndim == 4: 
        xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)  # B S H D -> B S H D/2 1 2
        xk_ = xk.reshape(*xk.shape[:-1], -1, 1, 2)  # B S H D -> B S H D/2 1 2
        # B S D/2 2 2 -> B S 1 D/2 2 2
        freqs_cis = freqs_cis.unsqueeze(seq_dim)
        xq_out = (xq_ * freqs_cis).sum(5).flatten(3)
        xk_out = (xk_ * freqs_cis).sum(5).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)
    elif xq.ndim == 3:
         # Expect freqs_cis shape (T, D/2, 2, 2)
        head_dim_idx = 1 # The Head dimension index
        flatten_start_dim = 2
        xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)  # T H D -> T H D/2 1 2
        xk_ = xk.reshape(*xk.shape[:-1], -1, 1, 2)  # T H D -> T H D/2 1 2
        freqs_cis_ = freqs_cis.unsqueeze(head_dim_idx)
        # xq_ * freqs_cis_ -> broadcasts to (B, S, H, D/2, 2, 2) or (T, H, D/2, 2, 2)
        # .sum(-1) -> sums the last dim -> (B, S, H, D/2, 2) or (T, H, D/2, 2)
        # .flatten(flatten_start_dim) -> reshapes to target (B, S, H, D) or (T, H, D)
        xq_out = (xq_ * freqs_cis_).sum(-1).flatten(flatten_start_dim)
        xk_out = (xk_ * freqs_cis_).sum(-1).flatten(flatten_start_dim)

        return xq_out.type_as(xq), xk_out.type_as(xk)

    raise ValueError(f"xq.ndim: {xq.ndim}, xk.ndim: {xk.ndim}")

def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def lengths_to_start_ids(lengths):
    doc_start = lengths.cumsum(0)
    doc_start = doc_start.roll(1)
    doc_start[0] = 0
    return doc_start


def lengths_to_local_ids(lengths):
    assert lengths.ndim == 1
    nb_seqs = lengths.size(0)
    total_seqlen = lengths.sum()
    # This gives the document id of each token
    doc_id = torch.repeat_interleave(lengths)
    # Compute document start for each document
    doc_start = lengths_to_start_ids(lengths)
    # Compute document start for each token
    doc_start = doc_start[doc_id]
    # Compute the position of each token within each document
    tok_id = torch.arange(total_seqlen, device=lengths.device) - doc_start

    return doc_id, tok_id


def generate_doc_mask_mod(
    mask_mod: _mask_mod_signature,
    lengths: torch.Tensor,
    kv_lengths: Optional[torch.Tensor] = None,
) -> _mask_mod_signature:
    """Generates mask mods that apply to inputs to flex attention in the sequence stacked
    format.

    Args:
        mask_mod: The mask mod to apply to the documents
        lengths: Lengths of each document

    Note:
        What is the sequence stacked format? When assembling batches of inputs, we
        take multiple sequences and stack them together to form 1 large sequence. We then
        use masking to ensure that the attention scores are only applied to tokens within
        the same document.

    Example:

    - Square mask
      doc_mask         lengths
      a a b b b c c    2 3 2
    a 1 0 0 0 0 0 0
    a 1 1 0 0 0 0 0
    b 0 0 1 0 0 0 0
    b 0 0 1 1 0 0 0
    b 0 0 1 1 1 0 0
    c 0 0 0 0 0 1 0
    c 0 0 0 0 0 1 1

    """
    kv_lengths = kv_lengths if kv_lengths is not None else lengths
    q_document_id, q_token_id = lengths_to_local_ids(lengths)
    kv_document_id, kv_token_id = lengths_to_local_ids(kv_lengths)
    q_max_idx = lengths.sum() - 1
    kv_max_idx = kv_lengths.sum() - 1

    def doc_mask_mod(b, h, q_idx, kv_idx):
        q_idx_cap = torch.minimum(q_max_idx, q_idx)
        kv_idx_cap = torch.minimum(kv_max_idx, kv_idx)
        valid_idx = (q_idx <= q_max_idx) & (kv_idx <= kv_max_idx)
        same_doc = q_document_id[q_idx_cap] == kv_document_id[kv_idx_cap]
        q_logical = q_token_id[q_idx_cap]
        kv_logical = kv_token_id[kv_idx_cap]
        inner_mask = mask_mod(b, h, q_logical, kv_logical)
        return same_doc & inner_mask & valid_idx

    return doc_mask_mod

@torch.compile
def modulate_and_gate(x, scale, gate):
    return (x * (1 + scale.unsqueeze(1))) * gate.unsqueeze(1).tanh()

@torch.compile
def modulate_and_gate_unpadded(x, scale, gate):
    return (x * (1 + scale)) * gate.tanh()


def create_causal_mask(seqlen, attn_impl):
    if attn_impl == "xformers":
        return fmha.attn_bias.LowerTriangularMask()
    elif attn_impl == "sdpa":
        return "causal"
    elif attn_impl == "flex_attention":
        return create_block_mask(causal_mask, None, None, seqlen, seqlen)
    else:
        raise NotImplementedError(f"Attention {attn_impl} is not implemented")


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)  # noqa: E741
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


class RMSNorm(nn.Module):
    """
    Initialize the RMSNorm normalization layer.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    """

    def __init__(self, dim: int, eps: float = 1e-6, liger_rms_norm: bool = True):
        super().__init__()
        self.eps = eps
        self.liger_rms_norm: bool = liger_rms_norm
        if liger_rms_norm:
            #  casting gemma: where everything is cast to fp32, then computed, then cast back to the original dtype.
            #  casting llama: where only the inverse RMS is computed on fp32.
            self.rms_norm = LigerRMSNorm(dim, init_fn="ones", eps=self.eps, casting_mode="llama")
        else:
            self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        if self.liger_rms_norm:
            return self.rms_norm(x)
        else:
            output = self._norm(x.float())
            return (output * self.weight.float()).type_as(x)

    def reset_parameters(self):
        if self.liger_rms_norm:
            torch.nn.init.ones_(self.rms_norm.weight)
        else:
            torch.nn.init.ones_(self.weight)  # type: ignore


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        rope_theta: float,
        qk_norm: bool = True,
        liger_rotary_emb: bool = True,
        liger_rms_norm: bool = True,
    ):
        super().__init__()

        self.dim = dim
        self.head_dim = head_dim
        self.rope_theta = rope_theta

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.heads_per_group = self.n_heads // self.n_kv_heads

        self.liger_rotary_emb = liger_rotary_emb

        self.wq = nn.Linear(
            dim,
            n_heads * head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            dim,
            n_kv_heads * head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            dim,
            n_kv_heads * head_dim,
            bias=False,
        )

        self.wo = nn.Linear(
            n_heads * head_dim,
            dim,
            bias=False,
        )

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, liger_rms_norm=liger_rms_norm)
            self.k_norm = RMSNorm(head_dim, liger_rms_norm=liger_rms_norm)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        attn_impl: str = "sdpa",
    ) -> torch.Tensor:
        # B S D
        bsz, seq_len, dim = x.shape
        xq = self.wq(x.view_as(x))
        xk = self.wk(x.view_as(x))
        xv = self.wv(x.view_as(x))

        output_shape = xq.shape
        # B S D -> B S H D
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq, xk = apply_rotary_emb(xq, xk, 2, freqs_cis[:, 0:seq_len], liger_rotary_emb=self.liger_rotary_emb)

        # This condition helps us be easily compatible
        # with inference by adding a pluggable KVCache
        if hasattr(self, "kv_cache"):
            xk, xv = self.kv_cache.update(xk, xv, tok_idx)

        xk = repeat_kv(xk, self.heads_per_group, dim=2)
        xv = repeat_kv(xv, self.heads_per_group, dim=2)

        if attn_impl == "flex_attention":
            assert mask is None or isinstance(mask, BlockMask)
            xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
            output = flex_attention_comp(xq, xk, xv, block_mask=mask)
            output = output.transpose(1, 2).contiguous()  # B H S D -> B S H D

        elif attn_impl == "fmha":
            assert mask is None or isinstance(mask, AttentionBias)
            output = fmha.memory_efficient_attention(xq, xk, xv, attn_bias=mask)
            # This uses B S H D instead of B H S D of pytorch

        elif attn_impl == "sdpa":
            xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
            assert mask is None or isinstance(mask, (str, torch.Tensor))
            is_causal = (mask == "causal") if isinstance(mask, str) else False
            mask = mask if isinstance(mask, torch.Tensor) else None
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                is_causal=is_causal,
                attn_mask=mask,
            )
            output = output.transpose(1, 2).contiguous()  # B H S D -> B S H D
        else:
            raise NotImplementedError(
                f"Attention implementation {attn_impl} not supported"
            )

        output = self.wo(output.reshape(output_shape))

        return output

    def reset_parameters(self):
        layer_init_kaiming_normal(self.wq)
        layer_init_kaiming_normal(self.wk)
        layer_init_kaiming_normal(self.wv)
        layer_init_kaiming_normal(self.wo)
        if isinstance(self.q_norm, RMSNorm):
            self.q_norm.reset_parameters()
        if isinstance(self.k_norm, RMSNorm):
            self.k_norm.reset_parameters()


class FlashAttention(nn.Module):
    """Multi-head attention module."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: Optional[int],
        qk_norm: bool,
        liger_rms_norm: bool = True,
        liger_rotary_emb: bool = True,
        window_size: Tuple[int, int] = (-1, -1),
        **kwargs,
    ):
        """
        Initialize the Attention module.

        Args:
            dim (int): Number of input dimensions.
            n_heads (int): Number of heads.
            n_kv_heads (Optional[int]): Number of kv heads, if using GQA.

        """
        super().__init__()
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.n_local_heads = n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = dim // n_heads
        self.liger_rotary_emb = liger_rotary_emb
        self.liger_rms_norm = liger_rms_norm    
        self.window_size = window_size
        self.qk_norm = qk_norm

        self.wq = nn.Linear(
            dim,
            n_heads * self.head_dim,
            bias=False,
        )  # mup
        self.wk = MuReadout(
            dim,
            n_kv_heads * self.head_dim,
            bias=False,
        )  # mup
        self.wv = MuReadout(
            dim,
            n_kv_heads * self.head_dim,
            bias=False,
        )  # mup

        self.wo = nn.Linear(
            n_heads * self.head_dim,
            dim,
            bias=False,
        )  # mup

        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, liger_rms_norm=liger_rms_norm)
            self.k_norm = RMSNorm(self.head_dim, liger_rms_norm=liger_rms_norm)
        else:
            self.q_norm = self.k_norm = nn.Identity()
        
        self.reset_parameters()

    def reset_parameters(self, *args, **kwargs):
        layer_init_kaiming_normal(self.wq)
        # MuReadout layers have their own initialization
        layer_init_kaiming_normal(self.wo)
        if self.qk_norm:
            self.q_norm.reset_parameters()
            self.k_norm.reset_parameters()

    # copied from huggingface modeling_llama.py
    def _upad_input(
        self, query_layer, key_layer, value_layer, attention_mask, query_length
    ):
        def _get_unpad_data(attention_mask):
            seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
            indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
            max_seqlen_in_batch = seqlens_in_batch.max().item()
            cu_seqlens = F.pad(
                torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
            )
            return (
                indices,
                cu_seqlens,
                max_seqlen_in_batch,
            )

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(
                    batch_size * kv_seq_len, self.n_local_heads, head_dim
                ),
                indices_k,
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
                query_layer, attention_mask
            )

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """

        Args:
            x:
            x_mask:
            freqs_cis:

        Returns:

        """
        dtype = x.dtype

        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        if cu_seqlens is None:
            bsz, seqlen, _ = x.shape
            xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
            xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
            xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        else:
            # padded case total_tokens, dim
            xq = xq.view(-1, self.n_local_heads, self.head_dim)
            xk = xk.view(-1, self.n_local_kv_heads, self.head_dim)
            xv = xv.view(-1, self.n_local_kv_heads, self.head_dim)

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)
        if cu_seqlens is None:
            xq, xk = apply_rotary_emb(xq, xk, 2, freqs_cis[:, 0:seqlen], liger_rotary_emb=self.liger_rotary_emb)
        else:
            xq, xk = apply_rotary_emb(xq, xk, 2, freqs_cis, liger_rotary_emb=False)
        xq, xk = xq.to(dtype), xk.to(dtype)

        softmax_scale = math.sqrt(1 / self.head_dim)

        if dtype in [torch.float16, torch.bfloat16]:
            if cu_seqlens is not None:
                output, _ = flash_attn_varlen_func(
                    xq,
                    xk,
                    xv,
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_k=max_seqlen,
                    causal=False,
                    softmax_scale=softmax_scale,
                    window_size=self.window_size,
                )
            else:
            # begin var_len flash attn
                (
                    query_states,
                    key_states,
                    value_states,
                    indices_q,
                    cu_seq_lens,
                    max_seq_lens,
                ) = self._upad_input(xq, xk, xv, x_mask, seqlen)

                cu_seqlens_q, cu_seqlens_k = cu_seq_lens
                max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens
                attn_output_unpad, _ = flash_attn_varlen_func(
                   query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    softmax_scale=softmax_scale,
                    causal=False,
                    window_size=self.window_size,
                )
                # print("OUTPUT", attn_output_unpad)
                output = pad_input(attn_output_unpad, indices_q, bsz, seqlen)
                # end var_len_flash_attn

        else:
            n_rep = self.n_local_heads // self.n_local_kv_heads
            if n_rep >= 1:
                xk = xk.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
                xv = xv.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            output = (
                F.scaled_dot_product_attention(
                    xq.permute(0, 2, 1, 3),
                    xk.permute(0, 2, 1, 3),
                    xv.permute(0, 2, 1, 3),
                    attn_mask=x_mask.bool()
                    .view(bsz, 1, 1, seqlen)
                    .expand(-1, self.n_local_heads, seqlen, -1),
                    scale=softmax_scale,
                )
                .permute(0, 2, 1, 3)
                .to(dtype)
            )

        output = output.flatten(-2)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        mp_size: int = 1,
        liger_ffn: bool = True,
    ):
        super().__init__()

        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        assert hidden_dim % mp_size == 0

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.liger_ffn = liger_ffn

        if liger_ffn:
            config = SimpleNamespace(
                hidden_size=dim,
                intermediate_size=hidden_dim,
                hidden_act="silu",
            )  # mup
            self.ffn = LigerSwiGLUMLP(config)
        else:
            self.w1 = nn.Linear(
                dim,
                hidden_dim,
                bias=False,
            )  # mup
            self.w3 = nn.Linear(
                dim,
                hidden_dim,
                bias=False,
            )  # mup
            self.w2 = nn.Linear(
                hidden_dim,
                dim,
                bias=False,
            )  # mup

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B S D
        if self.liger_ffn:
            return self.ffn(x)
        
        x1 = self.w1(x.view_as(x))
        x3 = self.w3(x.view_as(x))
        output = self.w2(F.silu(x1) * x3)
        return output

    def reset_parameters(self):
        if self.liger_ffn:
            layer_init_kaiming_normal(self.ffn.gate_proj)
            layer_init_kaiming_normal(self.ffn.up_proj)
            layer_init_kaiming_normal(self.ffn.down_proj)
        else:
            layer_init_kaiming_normal(self.w1)
            layer_init_kaiming_normal(self.w3)
            layer_init_kaiming_normal(self.w2)


class TransformerBlock(nn.Module):
    def __init__(self, args: BaseTransformerArgs):
        super().__init__()

        assert (args.head_dim is not None) or (
            args.n_heads is not None
        ), "Should specify at least head_dim or n_heads"
        self.head_dim = args.head_dim or args.dim // args.n_heads
        self.n_heads = args.n_heads or args.dim // self.head_dim
        self.n_kv_heads = args.n_kv_heads or self.n_heads

        assert args.n_heads % self.n_kv_heads == 0
        assert args.dim % args.n_heads == 0

        self.attention = Attention(
            dim=args.dim,
            head_dim=self.head_dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            rope_theta=args.rope_theta,
            qk_norm=args.qk_norm,
        )
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            liger_ffn=args.liger_ffn,
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps, liger_rms_norm=args.liger_rms_norm)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps, liger_rms_norm=args.liger_rms_norm)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, str]] = None,
        attn_impl: str = "sdpa",
    ) -> torch.Tensor:

        h = x + self.attention(
            self.attention_norm(x),
            freqs_cis,
            tok_idx=tok_idx,
            mask=mask,
            attn_impl=attn_impl,
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def init_weights(self):
        self.attention.reset_parameters()
        self.attention_norm.reset_parameters()

        self.feed_forward.reset_parameters()
        self.ffn_norm.reset_parameters()


class RotaryEmbedding1D(torch.nn.Module):
    """
    RotaryEmbedding Module
    """

    def __init__(self, theta: float, head_dim: int, max_seqlen: int = 1024):
        super().__init__()

        self.theta = theta
        self.head_dim = head_dim
        self.max_seqlen = max_seqlen

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(dim=head_dim, end=max_seqlen, theta=theta),
            persistent=False,
        )

    def reset_parameters(self):
        self.freqs_cis[...] = precompute_freqs_cis(
            dim=self.head_dim, end=self.max_seqlen, theta=self.theta
        )

    def forward(
        self, seqlen: Optional[int] = None, tok_idx: Optional[torch.Tensor] = None
    ):
        """
        Return freqs_cis corresponding to consecutive seqlen positions or the corresponding tok_idx positions
        Args:
            seqlen (int): Contiguous sequence length
            tok_idx (torch.Tensor[int]): Position indices of each token this overrides seqlen

        Returns:
            Tuple(torch.Tensor, torch.Tensor): Embedded input tensor and freqs_cis
        """
        test = (seqlen is not None) or (tok_idx is not None)
        assert test, "Should provide atleast seqlen or tok_idx"
        if tok_idx is not None:
            return self.freqs_cis[tok_idx]
        elif seqlen is not None:
            return self.freqs_cis[0:seqlen]


class RotaryEmbedding2D(torch.nn.Module):
    """
    RotaryEmbedding Module
    """

    def __init__(self, theta: float, head_dim: int, max_seqlen: int = 1024):
        super().__init__()

        self.theta = theta
        self.head_dim = head_dim
        self.max_seqlen = max_seqlen

        self.register_buffer(
            "freqs_cis",
            precompute_2d_freqs_cls(dim=head_dim, end=max_seqlen, theta=theta),
            persistent=False,
        )

    def reset_parameters(self):
        self.freqs_cis[...] = precompute_2d_freqs_cls(
            dim=self.head_dim, end=self.max_seqlen, theta=self.theta
        )

    def forward(
        self, seqlen: Optional[int] = None, tok_idx: Optional[torch.Tensor] = None
    ):
        """
        Return freqs_cis corresponding to consecutive seqlen positions or the corresponding tok_idx positions
        Args:
            seqlen (int): Contiguous sequence length
            tok_idx (torch.Tensor[int]): Position indices of each token this overrides seqlen

        Returns:
            Tuple(torch.Tensor, torch.Tensor): Embedded input tensor and freqs_cis
        """
        test = (seqlen is not None) or (tok_idx is not None)
        assert test, "Should provide atleast seqlen or tok_idx"
        if tok_idx is not None:
            return self.freqs_cis[tok_idx]
        elif seqlen is not None:
            return self.freqs_cis[0:seqlen]


class AdaLN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
    ):
        super().__init__()
        self.w1 = nn.Linear(
            in_dim,
            out_dim,
            bias=False,
        )
        self.in_dim = in_dim

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.w1(F.silu(x))
        return output

    def reset_parameters(self):
        layer_init_kaiming_normal(self.w1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size: int, time_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(
                time_embedding_size,
                hidden_size,
                bias=True,
            ),  # mup: input weights
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),  # mup: hidden weights
        )
        self.w2 = nn.Linear(
            hidden_size,
            hidden_size,
            bias=True,
        )
        self.hidden_size = hidden_size
        self.time_embedding_size = time_embedding_size

        self.reset_parameters()

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.time_embedding_size)
        t_emb = self.mlp(t_freq.to(self.w1.weight.dtype))
        return t_emb

    def reset_parameters(self):
        layer_init_kaiming_normal(self.mlp[0])
        layer_init_kaiming_normal(self.mlp[2])


class ImageEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.w1 = nn.Linear(
            in_features=in_dim,
            out_features=out_dim,
            bias=True,
        )  # mup: input weights
        self.in_dim = in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w1(x)

    def reset_parameters(self):
        layer_init_kaiming_normal(self.w1)
    