"""Student model: baseline GPT with RoPE instead of learned absolute positions."""
import torch
from torch import nn
from torch.nn import functional as F


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_len=256, base=10000.0):
        super().__init__()
        if head_dim % 2:
            raise ValueError('RoPE head dim must be even.')
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        freqs = torch.outer(torch.arange(max_len, dtype=torch.float32), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def forward(self, q, k):
        length = q.shape[-2]
        cos = self.cos_cached[:length][None, None].to(dtype=q.dtype)
        sin = self.sin_cached[:length][None, None].to(dtype=q.dtype)
        return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


def make_norm(width, rmsnorm):
    return RMSNorm(width) if rmsnorm else nn.LayerNorm(width)


class SwiGLU(nn.Module):
    def __init__(self, width, expansion=4):
        super().__init__()
        hidden = max(2, (int(round(width * expansion * 2 / 3)) // 2) * 2)
        self.gate_up = nn.Linear(width, 2 * hidden)
        self.down = nn.Linear(hidden, width)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class CausalDWConv(nn.Module):
    def __init__(self, width, kernel=3):
        super().__init__()
        self.kernel = kernel
        self.conv = nn.Conv1d(width, width, kernel_size=kernel, groups=width)

    def forward(self, x):
        return self.conv(F.pad(x.transpose(1, 2), (self.kernel - 1, 0))).transpose(1, 2)


class Block(nn.Module):
    def __init__(self, width=128, heads=4, swiglu=False, rmsnorm=False, qk_norm=False, gated_attn=False, short_conv=False,
                 dropout=0.0):
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.norm1, self.norm2 = make_norm(width, rmsnorm), make_norm(width, rmsnorm)
        self.short_conv = CausalDWConv(width) if short_conv else None
        self.qkv, self.proj = nn.Linear(width, 3 * width), nn.Linear(width, width)
        self.attn_gate = nn.Linear(width, width) if gated_attn else None
        self.mlp = SwiGLU(width) if swiglu else nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))
        self.q_norm = RMSNorm(width // heads) if qk_norm else None
        self.k_norm = RMSNorm(width // heads) if qk_norm else None

    def forward(self, x, rope):
        batch, length, width = x.shape
        hidden = self.norm1(x)
        if self.short_conv is not None:
            hidden = hidden + self.short_conv(hidden)
        q, k, v = self.qkv(hidden).view(batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = rope(q, k)
        attn_dropout = self.dropout if self.training else 0.0
        attended = self.proj(F.scaled_dot_product_attention(q, k, v, dropout_p=attn_dropout, is_causal=True)
                             .transpose(1, 2).reshape(batch, length, width))
        if self.attn_gate is not None:
            attended = attended * torch.sigmoid(self.attn_gate(hidden))
        x = x + F.dropout(attended, self.dropout, self.training)
        return x + F.dropout(self.mlp(self.norm2(x)), self.dropout, self.training)


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        self.use_pointer = bool(config.get('pointer', False))
        self.dropout = float(config.get('dropout', 0.0))
        width = config['width']
        self.token = nn.Embedding(config['vocab'], width)
        self.rope = RotaryEmbedding(width // config['heads'], max_len=self.context)
        self.blocks = nn.ModuleList([
            Block(width, config['heads'], swiglu=bool(config.get('swiglu', False)),
                  rmsnorm=bool(config.get('rmsnorm', False)), qk_norm=bool(config.get('qk_norm', False)),
                  gated_attn=bool(config.get('gated_attn', False)),
                  short_conv=bool(config.get('short_conv', False)), dropout=self.dropout)
            for _ in range(config['depth'])])
        self.norm = make_norm(width, bool(config.get('rmsnorm', False)))
        self.head = nn.Linear(width, config['vocab'], bias=False)
        if self.use_pointer:
            self.copy_q = nn.Linear(width, width)
            self.copy_k = nn.Linear(width, width)
            self.gate = nn.Linear(width, 1)
        self.apply(self.initialize)
        self.head.weight = self.token.weight
        if self.use_pointer:
            nn.init.constant_(self.gate.bias, 1.0)

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        x = F.dropout(self.token(ids), self.dropout, self.training)
        for block in self.blocks:
            x = block(x, self.rope)
        return self.norm(x)

    def _pointer_log_probs(self, ids):
        hidden = self.features(ids)
        vocab_logp = F.log_softmax(self.head(hidden).float(), dim=-1)
        generate = torch.sigmoid(self.gate(hidden)).float()
        query, key = self.copy_q(hidden).float(), self.copy_k(hidden).float()
        scores = torch.matmul(query, key.transpose(-1, -2)) * (query.shape[-1] ** -0.5)
        causal = torch.ones(ids.shape[1], ids.shape[1], dtype=torch.bool, device=ids.device).tril()
        scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
        copy_attn = torch.softmax(scores, dim=-1)
        copy_prob = vocab_logp.new_zeros(vocab_logp.shape)
        copy_prob.scatter_add_(2, ids.unsqueeze(1).expand_as(copy_attn), copy_attn)
        mix = generate * vocab_logp.exp() + (1 - generate) * copy_prob
        return mix.clamp_min(1e-12).log()

    def forward(self, ids):
        """Training interface: unnormalized next-token logits [batch, time, vocab]."""
        if self.use_pointer:
            return self._pointer_log_probs(ids)
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        """Evaluation interface: normalized log probabilities, with no access to targets."""
        if self.use_pointer:
            return self._pointer_log_probs(ids)
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    return GPT(config)
