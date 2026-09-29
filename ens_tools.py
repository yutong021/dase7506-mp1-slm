import argparse
import math
from pathlib import Path
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from common import load_data, make_model, windows

CACHE = Path('runs/ens_cache')


def cache(runs, device):
    CACHE.mkdir(parents=True, exist_ok=True)
    tokens, _ = load_data()['validation']
    for run in runs:
        out_path = CACHE / f'{run}.pt'
        if out_path.exists():
            continue
        saved = torch.load(f'runs/{run}/checkpoint.pt', map_location='cpu')
        model, _ = make_model(saved['implementation'], saved['config'], device)
        model.load_state_dict(saved['model'])
        model.eval()
        out = []
        with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
            for x, y in windows(tokens):
                lp = model.predict_log_probs(x.to(device)).float()
                y = y.to(device)
                mask = y >= 0
                out.append(lp.gather(-1, y.clamp(min=0)[..., None])[..., 0][mask].cpu())
        torch.save(torch.cat(out).double(), out_path)
        print(f'cached {run}', flush=True)
        del model
        torch.cuda.empty_cache()


def train_cache(runs, device, windows_count=256):
    CACHE.mkdir(parents=True, exist_ok=True)
    tokens, _ = load_data()['train']
    starts = torch.linspace(0, len(tokens) - 258, windows_count).long()
    batch = torch.stack([tokens[s:s + 257] for s in starts.tolist()])
    x, y = batch[:, :-1], batch[:, 1:]
    for run in runs:
        out_path = CACHE / f'train_{run}.pt'
        if out_path.exists():
            continue
        saved = torch.load(f'runs/{run}/checkpoint.pt', map_location='cpu')
        model, _ = make_model(saved['implementation'], saved['config'], device)
        model.load_state_dict(saved['model'])
        model.eval()
        out = []
        with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
            for i in range(0, len(x), 32):
                lp = model.predict_log_probs(x[i:i + 32].to(device)).float()
                out.append(lp.gather(-1, y[i:i + 32, :, None].to(device))[..., 0].flatten().cpu())
        torch.save(torch.cat(out).double(), out_path)
        print(f'cached train {run}', flush=True)
        del model
        torch.cuda.empty_cache()


def em(logp, iters=300):
    w = torch.full((logp.shape[0],), 1 / logp.shape[0], dtype=torch.double)
    for _ in range(iters):
        post = torch.softmax(logp + w.log()[:, None], 0)
        w = post.mean(1)
    return w


def nll(logp, w):
    return -torch.logsumexp(logp + w.clamp_min(1e-300).log()[:, None], 0).sum().item()


def select(names, nbytes, max_models):
    data = {n: torch.load(CACHE / f'{n}.pt') for n in names}
    n_tok = next(iter(data.values())).numel()
    to_bpb = lambda value, count: value / math.log(2) / (nbytes * count / n_tok)
    full = torch.stack([data[n] for n in names])
    print('single models (full validation BPB):')
    for n in sorted(names, key=lambda n: -data[n].sum().item()):
        print(f'  {to_bpb(-data[n].sum().item(), n_tok):.4f}  {n}')
    w = em(full)
    print(f'\nEM over all {len(names)} models: {to_bpb(nll(full, w), n_tok):.4f}')
    for n, weight in sorted(zip(names, w.tolist()), key=lambda t: -t[1]):
        print(f'  {weight:.4f}  {n}')
    half = n_tok // 2
    splits = {'A->B': (slice(0, half), slice(half, None)), 'B->A': (slice(half, None), slice(0, half))}
    for label, (fit, test) in splits.items():
        chosen = []
        print(f'\ngreedy selection, fit {label[0]}, evaluate {label[-1]}:')
        for _ in range(min(max_models, len(names))):
            best = None
            for n in names:
                if n in chosen:
                    continue
                logp = torch.stack([data[m][fit] for m in chosen + [n]])
                wk = em(logp, 100)
                value = nll(logp, wk)
                if best is None or value < best[0]:
                    best = (value, n)
            chosen.append(best[1])
            fit_logp = torch.stack([data[m][fit] for m in chosen])
            wk = em(fit_logp)
            test_logp = torch.stack([data[m][test] for m in chosen])
            n_test = test_logp.shape[1]
            print(f'  K={len(chosen):2d} fit {to_bpb(nll(fit_logp, wk), fit_logp.shape[1]):.4f} '
                  f'held-out {to_bpb(nll(test_logp, wk), n_test):.4f}  +{best[1]}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('cache', 'select', 'weights', 'train-cache'))
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-models', type=int, default=12)
    args = parser.parse_args()
    if args.mode == 'cache':
        cache(args.runs, torch.device(args.device))
    elif args.mode == 'train-cache':
        train_cache(args.runs, torch.device(args.device))
    elif args.mode == 'select':
        select(args.runs, load_data()['validation'][1], args.max_models)
    else:
        logp = torch.stack([torch.load(CACHE / f'{n}.pt') for n in args.runs])
        w = em(logp)
        nbytes = load_data()['validation'][1]
        print(f'EM BPB {nll(logp, w) / math.log(2) / nbytes:.4f}')
        print(' '.join(f'{x:.4f}' for x in w.tolist()))


if __name__ == '__main__':
    main()
