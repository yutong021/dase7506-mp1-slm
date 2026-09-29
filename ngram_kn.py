"""Interpolated modified Kneser-Ney n-gram LM over the fixed BPE-2048 tokens (teacher-ensemble candidate).

Contexts never cross the scored 256-target windows. Writes validation target log-probs aligned with
common.windows() so they can be mixed with neural models' target log-probs.
"""
import argparse
import math
from pathlib import Path
import numpy as np
import torch
from common import load_data, windows

V = 2048


def ngram_keys(tokens, n):
    keys = np.zeros(len(tokens) - n + 1, dtype=np.int64)
    for i in range(n):
        keys = keys * V + tokens[i:len(tokens) - n + 1 + i]
    return keys


def discounts(counts):
    n = [np.sum(counts == k) for k in (1, 2, 3, 4)]
    y = n[0] / (n[0] + 2 * n[1])
    return np.array([0., 1 - 2 * y * n[1] / n[0], 2 - 3 * y * n[2] / n[1], 3 - 4 * y * n[3] / n[2]])


class Table:
    """Counts of n-grams plus per-context totals and discount mass (for n >= 2)."""

    def __init__(self, keys, counts, n):
        order = np.argsort(keys)
        self.keys, self.counts, self.n = keys[order], counts[order].astype(np.float64), n
        self.d = discounts(self.counts)
        if n == 1:
            self.total = self.counts.sum()
            self.mass = (self.d[np.minimum(self.counts, 3).astype(int)]).sum()
            return
        ctx = self.keys // V
        self.ctx_keys, start = np.unique(ctx, return_index=True)
        self.ctx_total = np.add.reduceat(self.counts, start)
        self.ctx_mass = np.add.reduceat(self.d[np.minimum(self.counts, 3).astype(int)], start)

    def lookup(self, keys):
        pos = np.clip(np.searchsorted(self.keys, keys), 0, len(self.keys) - 1)
        return np.where(self.keys[pos] == keys, self.counts[pos], 0.)

    def context(self, ctx):
        pos = np.clip(np.searchsorted(self.ctx_keys, ctx), 0, len(self.ctx_keys) - 1)
        hit = self.ctx_keys[pos] == ctx
        return np.where(hit, self.ctx_total[pos], 0.), np.where(hit, self.ctx_mass[pos], 0.)


def build(train, order):
    tables = []
    for n in range(1, order + 1):
        if n == order:
            keys, counts = np.unique(ngram_keys(train, n), return_counts=True)
        else:
            longer = np.unique(ngram_keys(train, n + 1))
            keys, counts = np.unique(longer % (V ** n), return_counts=True)
        tables.append(Table(keys, counts, n))
    return tables


def target_log_probs(tables, x, y):
    """x, y: [rows, 256] windows; returns log p(y) for every valid target, row-major."""
    order = len(tables)
    rows, length = x.shape
    t = np.arange(length)[None, :]
    target = np.where(y >= 0, y, 0)
    uni = tables[0]
    c = uni.lookup(target)
    p = (np.maximum(c - uni.d[np.minimum(c, 3).astype(int)], 0) + uni.mass / V) / uni.total
    for n in range(2, order + 1):
        m = n - 1
        valid = t - m + 1 >= 0
        ctx = np.zeros((rows, length), dtype=np.int64)
        for j in range(m):
            shifted = np.zeros_like(x)
            shift = m - 1 - j
            shifted[:, shift:] = x[:, :length - shift]
            ctx = ctx * V + shifted
        table = tables[n - 1]
        total, mass = table.context(ctx)
        c = table.lookup(ctx * V + target)
        discounted = np.maximum(c - table.d[np.minimum(c, 3).astype(int)], 0)
        seen = valid & (total > 0)
        p = np.where(seen, (discounted + mass * p) / np.maximum(total, 1), p)
    return np.log(p)[y >= 0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--orders', type=int, nargs='+', default=[3, 4, 5])
    ap.add_argument('--output-dir', type=Path, default=Path('runs/ens_cache'))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_data()
    train = data['train'][0].numpy().astype(np.int64)
    val, vbytes = data['validation']
    x, y = map(lambda parts: torch.cat(parts).numpy().astype(np.int64), zip(*windows(val)))
    for order in args.orders:
        lp = target_log_probs(build(train, order), x, y)
        torch.save(torch.from_numpy(lp).double(), args.output_dir / f'KN{order}.pt')
        print(f'KN{order}: val BPB {-lp.sum() / math.log(2) / vbytes:.4f}', flush=True)


if __name__ == '__main__':
    main()
