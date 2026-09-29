"""Summarize every runs/*/metrics.json (plus CPU FP32 test scores when present) into one CSV."""
import csv
import json
import sys
from pathlib import Path

FIELDS = ['run', 'implementation', 'parameters', 'steps', 'lr', 'ema_decay', 'dropout', 'teachers', 'distill_alpha',
          'oof_mix', 'holdout_fold', 'valid_bpb', 'test_bpb_cpu_fp32', 'test_cpu_seconds', 'train_seconds']


def row(run):
    m = json.loads((run / 'metrics.json').read_text())
    test_path = run / 'test_cpu_fp32.json'
    test = json.loads(test_path.read_text()) if test_path.exists() else {}
    oof = m.get('oof_teachers')
    teachers = len(m.get('teacher') or []) + (sum(len(g['checkpoints']) for g in oof['groups']) if oof else 0)
    return {'run': run.name, 'implementation': m['implementation'], 'parameters': m['parameters'],
            'steps': m['train_tokens'] // (32 * 256), 'lr': m.get('lr', 0.001), 'ema_decay': m.get('ema_decay', 0.0),
            'dropout': m['config'].get('dropout', 0.0), 'teachers': teachers, 'distill_alpha': m.get('distill_alpha', 0.0),
            'oof_mix': m.get('oof_mix', ''), 'holdout_fold': m.get('holdout_fold', ''),
            'valid_bpb': round(m['validation']['bpb'], 4),
            'test_bpb_cpu_fp32': round(test['bpb'], 4) if test else '',
            'test_cpu_seconds': round(test['seconds'], 2) if test else '',
            'train_seconds': round(m['train_seconds'])}


def main():
    runs = sorted(p.parent for p in Path('runs').glob('*/metrics.json'))
    writer = csv.DictWriter(sys.stdout, FIELDS)
    writer.writeheader()
    for run in runs:
        writer.writerow(row(run))


if __name__ == '__main__':
    main()
