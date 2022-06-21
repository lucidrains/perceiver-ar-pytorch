import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange

def exists(val):
    return val is not None

def FeedForward(dim, mult = 4, dropout = 0.):
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim, bias = False),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim, bias = False)
    )

class CausalAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = heads * dim_head

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        i, j = sim.shape[-2:]
        causal_mask = torch.ones((i, j), device = x.device, dtype = torch.bool).triu(j - i + 1)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class CausalPrefixAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = heads * dim_head

        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, context):
        x = self.norm(x)
        q = self.to_q(x)

        k_input, v_input = self.to_kv(x).chunk(2, dim = -1)
        k_context, v_context = self.to_kv(context).chunk(2, dim = -1)

        k = torch.cat((k_context, k_input), dim = 1)
        v = torch.cat((v_context, v_input), dim = 1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        i, j = sim.shape[-2:]
        causal_mask = torch.ones((i, j), device = x.device, dtype = torch.bool).triu(j - i + 1)

        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)

class PerceiverAR(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        max_seq_len,
        cross_attn_seq_len,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        ff_mult = 4
    ):
        super().__init__()
        assert max_seq_len > cross_attn_seq_len, 'max_seq_len must be greater than cross_attn_seq_len, the length of the sequence for which to cross attend to "perceiver" style'
        self.max_seq_len = max_seq_len
        self.cross_attn_seq_len = cross_attn_seq_len

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.perceive_layer = nn.ModuleList([
            CausalPrefixAttention(dim = dim, dim_head = dim_head, heads = heads, dropout = dropout),
            FeedForward(dim, mult = ff_mult, dropout = dropout)
        ])

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                CausalAttention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim, mult = ff_mult, dropout = dropout),
            ]))

        self.to_logits = nn.Linear(dim, num_tokens, bias = False)

    def forward(
        self,
        x,
        labels = None
    ):
        seq_len, device = x.shape[1], x.device
        assert self.cross_attn_seq_len < seq_len <= self.max_seq_len

        x = self.token_emb(x)
        x = x + self.pos_emb(torch.arange(seq_len, device = device))

        prefix, x = x[:, :self.cross_attn_seq_len], x[:, self.cross_attn_seq_len:]

        # initial perceiver attention and feedforward (one cross attention)

        cross_attn, ff = self.perceive_layer

        x = cross_attn(x, prefix) + x
        x = ff(x) + x

        # layers

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        # to logits

        logits = self.to_logits(x)

        # take care of cross entropy loss if labels are provided

        if not exists(labels):
            return logits

        labels = labels[:, self.cross_attn_seq_len:]
        return F.cross_entropy(rearrange(logits, 'b n c -> b c n'), labels, ignore_index = 0)
