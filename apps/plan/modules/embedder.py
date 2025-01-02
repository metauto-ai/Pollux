import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size: int, time_embedding_size: int = 256):
        super().__init__()
        self.w1 = nn.Linear(
            time_embedding_size,
            hidden_size,
            bias=True,
        )
        self.w2 = nn.Linear(
            hidden_size,
            hidden_size,
            bias=True,
        )
        self.hidden_size = hidden_size
        self.time_embedding_size = time_embedding_size

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
        t_emb = self.w1(t_freq.to(self.w1.weight.dtype))
        t_emb = self.w2(F.silu(t_emb))
        return t_emb

    def reset_parameters(self, init_std=None, factor=1.0):
        in_init_std = init_std or (self.time_embedding_size ** (-0.5))
        out_init_std = init_std or (self.hidden_size ** (-0.5))
        out_init_std = out_init_std / factor
        nn.init.trunc_normal_(
            self.w1.weight,
            mean=0.0,
            std=in_init_std,
            a=-3 * in_init_std,
            b=3 * in_init_std,
        )
        nn.init.trunc_normal_(
            self.w2.weight,
            mean=0.0,
            std=out_init_std,
            a=-3 * out_init_std,
            b=3 * out_init_std,
        )
        nn.init.normal_(self.w1.bias, std=0.02)
        nn.init.normal_(self.w2.bias, std=0.02)


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
        )
        self.in_dim = in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w1(x)

    def reset_parameters(self, init_std=None, factor=1.0):
        init_std = init_std or (self.in_dim ** (-0.5))
        init_std = init_std / factor
        nn.init.trunc_normal_(
            self.w1.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )

        nn.init.normal_(self.w1.bias, std=0.02)


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    https://github.com/facebookresearch/DiT/blob/main/models.py
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.hidden_size = hidden_size

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

    def reset_parameters(self, init_std=None, factor=1.0):
        init_std = init_std or (self.hidden_size ** (-0.5))
        init_std = init_std / factor
        nn.init.trunc_normal_(
            self.embedding_table.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )
