"""Teacher-only architectures for building a diverse distillation ensemble (never used as the scored student).

Select with config["arch"]:
  "gpt"  - the student GPT (same as student.py)
  "mos"  - GPT with a Mixture-of-Softmaxes output layer (config["mos"] components)
  "moe"  - GPT whose feed-forward layers are top-k routed SwiGLU experts
  "lstm" - AWD-LSTM style recurrent LM (embedding/locked/weight dropout, tied output)
Every architecture can use the within-window pointer/copy mixture via config["pointer"].
"""
import torch
from torch import nn
from torch.nn import functional as F
from student import GPT, SwiGLU


def pointer_mix(model, ids, hidden, vocab_logp):
    generate = torch.sigmoid(model.gate(hidden)).float()
    query, key = model.copy_q(hidden).float(), model.copy_k(hidden).float()
    scores = torch.matmul(query, key.transpose(-1, -2)) * (query.shape[-1] ** -0.5)
    causal = torch.ones(ids.shape[1], ids.shape[1], dtype=torch.bool, device=ids.device).tril()
    scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
    copy_attn = torch.softmax(scores, dim=-1)
    copy_prob = vocab_logp.new_zeros(vocab_logp.shape)
    copy_prob.scatter_add_(2, ids.unsqueeze(1).expand_as(copy_attn), copy_attn)
    mix = generate * vocab_logp.exp() + (1 - generate) * copy_prob
    return mix.clamp_min(1e-12).log()


def init_normal(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, std=.02)
        if getattr(module, 'bias', None) is not None:
            nn.init.zeros_(module.bias)


class ZooMixin:
    """Shared output path: normalized log-probs (optionally pointer-mixed) for both training and evaluation."""

    def vocab_log_probs(self, hidden):
        return F.log_softmax(self.head(hidden).float(), dim=-1)

    def log_probs(self, ids):
        hidden = self.features(ids)
        vocab_logp = self.vocab_log_probs(hidden)
        return pointer_mix(self, ids, hidden, vocab_logp) if self.use_pointer else vocab_logp

    def forward(self, ids):
        return self.log_probs(ids)

    def predict_log_probs(self, ids):
        return self.log_probs(ids)


class ZooGPT(ZooMixin, GPT):
    pass


class MoSGPT(ZooMixin, GPT):
    def __init__(self, config):
        super().__init__(config)
        width, self.components = config['width'], int(config['mos'])
        self.prior = nn.Linear(width, self.components, bias=False)
        self.latent = nn.Linear(width, self.components * width)
        self.prior.apply(init_normal)
        self.latent.apply(init_normal)

    def vocab_log_probs(self, hidden):
        batch, length, width = hidden.shape
        prior = F.log_softmax(self.prior(hidden).float(), dim=-1)
        latent = torch.tanh(self.latent(hidden)).view(batch, length, self.components, width)
        latent = F.dropout(latent, self.dropout, self.training)
        component_logp = F.log_softmax(self.head(latent).float(), dim=-1)
        return torch.logsumexp(prior.unsqueeze(-1) + component_logp, dim=2)


class MoE(nn.Module):
    def __init__(self, width, experts=8, top_k=2, expansion=2, aux_weight=0.01):
        super().__init__()
        self.top_k, self.aux_weight = top_k, aux_weight
        self.router = nn.Linear(width, experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(width, expansion) for _ in range(experts)])
        self.aux_loss = None

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        probs = torch.softmax(self.router(flat).float(), dim=-1)
        weight, index = probs.topk(self.top_k, dim=-1)
        weight = weight / weight.sum(-1, keepdim=True)
        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            rows, slot = (index == e).nonzero(as_tuple=True)
            if rows.numel():
                out.index_add_(0, rows, (expert(flat[rows]) * weight[rows, slot, None]).to(out.dtype))
        if self.training:
            load = F.one_hot(index[:, 0], len(self.experts)).float().mean(0)
            self.aux_loss = self.aux_weight * len(self.experts) * (load * probs.mean(0)).sum()
        else:
            self.aux_loss = None
        return out.view(shape)


class MoEGPT(ZooMixin, GPT):
    def __init__(self, config):
        super().__init__(config)
        for block in self.blocks:
            block.mlp = MoE(config['width'], experts=int(config.get('experts', 8)), top_k=int(config.get('top_k', 2)),
                            expansion=float(config.get('expert_expansion', 2)),
                            aux_weight=float(config.get('aux_weight', 0.01)))
            block.mlp.apply(init_normal)

    @property
    def aux_loss(self):
        losses = [block.mlp.aux_loss for block in self.blocks if block.mlp.aux_loss is not None]
        return sum(losses) if losses else None


def locked_dropout(x, p, training):
    if not training or p == 0:
        return x
    mask = x.new_empty(x.shape[0], 1, x.shape[2]).bernoulli_(1 - p) / (1 - p)
    return x * mask


class WeightDropLSTM(nn.LSTM):
    """DropConnect on the hidden-to-hidden matrices (AWD-LSTM)."""

    def __init__(self, *args, weight_dropout=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_dropout = weight_dropout

    def forward(self, x, hx=None):
        if not (self.training and self.weight_dropout):
            return super().forward(x, hx)
        flat = self._flat_weights
        self._flat_weights = [F.dropout(w, self.weight_dropout, True) if name.startswith('weight_hh') else w
                              for name, w in zip(self._flat_weights_names, flat)]
        try:
            return super().forward(x, hx)
        finally:
            self._flat_weights = flat


class LSTMLM(ZooMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        self.use_pointer = bool(config.get('pointer', False))
        width, hidden, depth = config['width'], config['hidden'], config['depth']
        self.dropout = float(config.get('dropout', 0.4))
        self.dropouti = float(config.get('dropouti', 0.4))
        self.dropouth = float(config.get('dropouth', 0.25))
        self.dropoute = float(config.get('dropoute', 0.1))
        self.token = nn.Embedding(config['vocab'], width)
        self.rnns = nn.ModuleList([
            WeightDropLSTM(width if layer == 0 else hidden, width if layer == depth - 1 else hidden,
                           batch_first=True, weight_dropout=float(config.get('wdrop', 0.5)))
            for layer in range(depth)])
        self.head = nn.Linear(width, config['vocab'], bias=False)
        if self.use_pointer:
            self.copy_q = nn.Linear(width, width)
            self.copy_k = nn.Linear(width, width)
            self.gate = nn.Linear(width, 1)
        for module in (self.token, self.head) + ((self.copy_q, self.copy_k, self.gate) if self.use_pointer else ()):
            module.apply(init_normal)
        # LSTM outputs are bounded by tanh and the output layer is tied, so a GPT-sized (0.02) embedding
        # leaves logits and recurrent gradients ~1e3x too small to train.
        nn.init.normal_(self.token.weight, std=float(config.get('emb_std', 0.1)))
        self.head.weight = self.token.weight
        if self.use_pointer:
            nn.init.constant_(self.gate.bias, 1.0)

    def features(self, ids):
        weight = self.token.weight
        if self.training and self.dropoute:
            keep = weight.new_empty(weight.shape[0], 1).bernoulli_(1 - self.dropoute) / (1 - self.dropoute)
            weight = weight * keep
        x = locked_dropout(F.embedding(ids, weight), self.dropouti, self.training)
        for layer, rnn in enumerate(self.rnns):
            x, _ = rnn(x)
            if layer < len(self.rnns) - 1:
                x = locked_dropout(x, self.dropouth, self.training)
        return locked_dropout(x, self.dropout, self.training)


ARCHS = {'gpt': ZooGPT, 'mos': MoSGPT, 'moe': MoEGPT, 'lstm': LSTMLM}


def build_model(config):
    return ARCHS[config.get('arch', 'gpt')](config)
