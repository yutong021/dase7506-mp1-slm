"""Default recipe: 1,200 steps x 32 sequences x 256 targets = 9,830,400 tokens."""
import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from common import PROTOCOL, ROOT, autocast, device_metrics, load_data, make_model, setup, sha
from evaluate import score


def exact_attention(device):
    # The FP32 memory-efficient SDPA kernel is non-deterministic on RTX 5090 / torch 2.7.1 (~0.1 output noise).
    return sdpa_kernel([SDPBackend.MATH]) if device.type == 'cuda' else nullcontext()


def update_ema(ema, model, decay):
    with torch.no_grad():
        for name, param in model.state_dict().items():
            if param.is_floating_point():
                ema[name].lerp_(param, 1 - decay)
            else:
                ema[name].copy_(param)


def swap_weights(model, weights):
    current = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    model.load_state_dict(weights)
    return current


def make_adamw(model, lr=0.001, weight_decay=0.1, grouped=False):
    if not grouped:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    decay, no_decay, seen = [], [], set()
    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        if param.ndim <= 1 or 'norm' in name.lower() or name.endswith('token.weight'):
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [{'params': decay, 'weight_decay': weight_decay},
         {'params': no_decay, 'weight_decay': 0.0}],
        lr=lr)


def fold_bounds(length, folds):
    return [length*i//folds for i in range(folds+1)]


def fold_window_starts(bounds, folds, span=257):
    """Starts of windows lying entirely inside one of the given contiguous folds."""
    return torch.cat([torch.arange(bounds[k], bounds[k+1]-span+1) for k in folds])


def load_teacher(path, device):
    saved = torch.load(path, map_location='cpu')
    teacher, _ = make_model(saved['implementation'], saved['config'], device)
    teacher.load_state_dict(saved['model'])
    return teacher.eval().requires_grad_(False)


def main():
    total_started = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation', default='student')
    p.add_argument('--config', type=Path, default=ROOT/'configs/baseline.json')
    p.add_argument('--run-dir', type=Path, default=ROOT/'runs/baseline-s17')
    p.add_argument('--device', default='cpu')
    p.add_argument('--precision', choices=['auto','fp32','bf16'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--eval-every', type=int, default=0,
                   help='Optional validation-curve interval; 0 evaluates only after training.')
    p.add_argument('--ema-decay', type=float, default=0.0,
                   help='If >0, validate and save EMA weights. Typical value: 0.999.')
    p.add_argument('--adamw-grouped', action='store_true',
                   help='Do not apply weight decay to embeddings, norms, or 1D parameters.')
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--teacher', type=Path, nargs='+', default=None,
                   help='Optional frozen teacher checkpoints; several are ensembled by averaging probabilities.')
    p.add_argument('--teacher-weights', type=float, nargs='+', default=None,
                   help='Optional mixture weights for --teacher (normalized to sum to 1); default is uniform.')
    p.add_argument('--distill-alpha', type=float, default=0.5,
                   help='Weight of the KL(teacher || student) term; the hard-label loss gets 1-alpha.')
    p.add_argument('--distill-temp', type=float, default=1.0)
    p.add_argument('--folds', type=int, default=0,
                   help='Split the training tokens into this many contiguous folds (used with --holdout-fold).')
    p.add_argument('--holdout-fold', type=int, default=None,
                   help='Train only on windows inside the other folds, e.g. for an out-of-fold teacher.')
    p.add_argument('--oof-teachers', type=Path, default=None,
                   help='JSON {"folds": K, "groups": [{"weight": w, "checkpoints": [K paths]}]}; '
                        'checkpoints[k] never saw fold k and labels only windows inside fold k.')
    p.add_argument('--oof-mix', type=float, default=0.5,
                   help='With both --teacher and --oof-teachers, probability weight of the out-of-fold teacher.')
    args = p.parse_args()
    if args.holdout_fold is not None and not 0 <= args.holdout_fold < args.folds:
        p.error('--holdout-fold needs --folds K >= 2 and 0 <= fold < K.')
    if args.oof_teachers is not None and args.holdout_fold is not None:
        p.error('--oof-teachers cannot be combined with --holdout-fold.')
    if not 0 <= args.oof_mix <= 1:
        p.error('--oof-mix must be in [0, 1].')
    if not 0 <= args.distill_alpha <= 1 or args.distill_temp <= 0:
        p.error('Distillation alpha must be in [0, 1] and temperature positive.')
    if args.teacher_weights is not None and (not args.teacher or len(args.teacher_weights) != len(args.teacher)
                                             or min(args.teacher_weights) <= 0):
        p.error('Teacher weights must be positive and match the number of teachers.')
    if args.ema_decay < 0 or args.ema_decay >= 1:
        p.error('EMA decay must be 0 (disabled) or in (0, 1).')
    if args.steps < 1 or args.batch_size < 1 or args.lr <= 0:
        p.error('Batch size, step count and learning rate must be positive.')
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        p.error('Run directory already contains results. Use a new --run-dir.')
    device, precision = setup(args.device, args.precision, args.threads)
    torch.manual_seed(args.seed)
    prepared = time.perf_counter()
    data = load_data()
    config = json.loads(args.config.read_text())
    model, implementation_sha = make_model(args.implementation, config, device)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    optimizer = make_adamw(model, lr=args.lr, grouped=args.adamw_grouped)
    ema = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()} if args.ema_decay else None
    train_length = len(data['train'][0])
    valid_starts, bounds, oof_spec = None, None, None
    if args.holdout_fold is not None:
        bounds = fold_bounds(train_length, args.folds)
        valid_starts = fold_window_starts(bounds, [k for k in range(args.folds) if k != args.holdout_fold])
    def log_mixture_weights(weights):
        weights = torch.tensor(weights, device=device)
        return (weights/weights.sum()).log()[:,None,None,None]
    full_teachers, oof_sets = [], []
    if args.teacher:
        full_teachers = [load_teacher(path, device) for path in args.teacher]
        full_log_weights = log_mixture_weights(args.teacher_weights or [1.]*len(full_teachers))
    if args.oof_teachers is not None:
        oof_spec = json.loads(args.oof_teachers.read_text())
        bounds = fold_bounds(train_length, oof_spec['folds'])
        valid_starts = fold_window_starts(bounds, range(oof_spec['folds']))
        # oof_sets[k] labels only the windows inside fold k.
        oof_sets = [[load_teacher(group['checkpoints'][k], device) for group in oof_spec['groups']]
                    for k in range(oof_spec['folds'])]
        oof_log_weights = log_mixture_weights([group['weight'] for group in oof_spec['groups']])
        inner_bounds = torch.tensor(bounds[1:-1], device=device)
    oof_mix = args.oof_mix if full_teachers and oof_sets else float(bool(oof_sets))
    tokens = data['train'][0].to(device)
    rng = torch.Generator().manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter()-prepared
    started = time.perf_counter()
    history = []
    validation_history = []
    intermediate_validation_seconds = 0.
    for step in range(args.steps):
        if valid_starts is None:
            starts = torch.randint(len(tokens)-257, (args.batch_size,), generator=rng).to(device)
        else:
            starts = valid_starts[torch.randint(len(valid_starts), (args.batch_size,), generator=rng)].to(device)
        batch = tokens[starts[:,None]+torch.arange(257,device=device)]
        learning_rate = args.lr * min(1.,(step+1)/100) * (.1+.9*.5*(1+math.cos(math.pi*step/args.steps)))
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            output = model(batch[:,:-1]).flatten(0,1).float()
            hard_loss = F.cross_entropy(output,batch[:,1:].flatten())
            loss = hard_loss
            if full_teachers or oof_sets:
                with torch.no_grad():
                    inputs = batch[:,:-1]
                    parts = []
                    if full_teachers and oof_mix < 1:
                        full = torch.logsumexp(torch.stack([t.predict_log_probs(inputs).float()
                                                            for t in full_teachers])+full_log_weights,0)
                        parts.append(full+math.log(1-oof_mix))
                    if oof_sets and oof_mix > 0:
                        fold_of = torch.bucketize(starts, inner_bounds, right=True)
                        oof = torch.empty(*inputs.shape, output.shape[-1], device=device)
                        for k, members in enumerate(oof_sets):
                            rows = (fold_of == k).nonzero()[:,0]
                            if rows.numel():
                                oof[rows] = torch.logsumexp(torch.stack([t.predict_log_probs(inputs[rows]).float()
                                                                         for t in members])+oof_log_weights,0)
                        parts.append(oof+math.log(oof_mix))
                    mixture = torch.logsumexp(torch.stack(parts),0)
                    target = F.log_softmax(mixture.flatten(0,1)/args.distill_temp,-1)
                soft_loss = F.kl_div(F.log_softmax(output/args.distill_temp,-1),target,log_target=True,
                                     reduction='batchmean')*args.distill_temp**2
                loss = (1-args.distill_alpha)*hard_loss+args.distill_alpha*soft_loss
            aux_loss = getattr(model, 'aux_loss', None)
            if aux_loss is not None:
                loss = loss+aux_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if ema is not None and (step + 1) > 100:
            update_ema(ema, model, args.ema_decay)
        if (step+1)%100 == 0 or step+1 == args.steps:
            row = {'step':step+1,'loss':loss.item(),'hard_loss':hard_loss.item(),'seconds':time.perf_counter()-started-intermediate_validation_seconds}
            history.append(row)
            print(json.dumps(row),flush=True)
        if args.eval_every > 0 and (step+1)%args.eval_every == 0:
            raw = swap_weights(model, ema) if ema is not None else None
            with exact_attention(device):
                intermediate = score(model,*data['validation'],device,'fp32')
            if raw is not None:
                model.load_state_dict(raw)
            intermediate.pop('window_nll_nats')
            intermediate_validation_seconds += intermediate['seconds']
            validation_history.append({'step':step+1,**intermediate})
            print(json.dumps({'validation':validation_history[-1]}),flush=True)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter()-started-intermediate_validation_seconds
    if ema is not None:
        model.load_state_dict(ema)
    with exact_attention(device):
        validation = score(model,*data['validation'],device,'fp32')
    validation.pop('window_nll_nats')
    checkpoint = args.run_dir/'checkpoint.pt'
    torch.save({'protocol':PROTOCOL,'implementation':args.implementation,'config':config,
                'model':model.cpu().state_dict(),'seed':args.seed,
                'train_tokens':args.steps*args.batch_size*256},checkpoint)
    result = {'protocol':PROTOCOL,'implementation':args.implementation,'config':config,'seed':args.seed,
              'ema_decay':args.ema_decay,'adamw_grouped':args.adamw_grouped,'lr':args.lr,
              'teacher':[str(path) for path in args.teacher] if args.teacher else None,
              'teacher_sha256':[sha(path) for path in args.teacher] if args.teacher else None,
              'teacher_weights':args.teacher_weights,
              'folds':args.folds,'holdout_fold':args.holdout_fold,'oof_teachers':oof_spec,'oof_mix':oof_mix,
              'distill_alpha':args.distill_alpha if full_teachers or oof_sets else 0.,'distill_temp':args.distill_temp,
              'parameters':sum(p.numel() for p in model.parameters()),'precision':precision,
              'train_tokens':args.steps*args.batch_size*256,'preparation_seconds':preparation_seconds,
              'train_seconds':train_seconds,'validation':validation,'history':history,
              'validation_history':validation_history,
              'intermediate_validation_seconds':intermediate_validation_seconds,
              'process_seconds':time.perf_counter()-total_started,
              'torch_version':str(torch.__version__),'threads':args.threads,
              'checkpoint_sha256':sha(checkpoint),'implementation_sha256':implementation_sha,
              **device_metrics(device)}
    (args.run_dir/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result|{'history':[]},indent=2),flush=True)


if __name__ == '__main__':
    main()
