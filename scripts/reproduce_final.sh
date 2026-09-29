#!/usr/bin/env bash
# Rebuilds the submitted checkpoint runs/kd_hyb_ens8w_oof4_do05_s40000_a07_m03 from scratch (seed 17, one GPU).
# Stages must run in order; runs inside a stage are independent and can be spread over several GPUs.
set -euo pipefail
cd "$(dirname "$0")/.."

train() {  # train <run-name> <config> <steps> <lr> [extra args...]
  local name="$1" config="$2" steps="$3" lr="$4"; shift 4
  [ -f "runs/$name/checkpoint.pt" ] && return
  python train.py --implementation student --config "configs/$config.json" --device cuda --seed 17 \
    --steps "$steps" --ema-decay 0.99 --lr "$lr" --run-dir "runs/$name" "$@"
}
ck() { for run in "$@"; do printf 'runs/%s/checkpoint.pt ' "$run"; done; }

# Stage 1: single teachers
train teacher_w384_d8_h4_do30_s12000_ema099_lr1e3 teacher_w384_d8_h4_do30 12000 0.001
train teacher_w384_d8_h4_do20_s8000_ema099_lr1e3  teacher_w384_d8_h4_do20 8000  0.001
train teacher_w384_d8_h4_do40_s16000_ema099_lr1e3 teacher_w384_d8_h4_do40 16000 0.001
train teacher_w512_d8_h4_do40_s12000_ema099_lr1e3 teacher_w512_d8_h4_do40 12000 0.001
train teacher_w384_d8_h4_do20_s20000_ema099_lr1e3 teacher_w384_d8_h4_do20 20000 0.001
train teacher_w512_d8_h4_do30_s20000_ema099_lr1e3 teacher_w512_d8_h4_do30 20000 0.001
train rope_w240_d6_h2_do20_s30000_ema099_lr12e4   rope_w240_d6_h2_do20    30000 0.0012
train rope_w240_d6_h2_do10_s12000_ema099_lr12e4   rope_w240_d6_h2_do10    12000 0.0012
train rope_w256_d6_h2_s2400_ema099                rope_w256_d6_h2         2400  0.001
train rope_w224_d6_gate_conv_s2400_ema099         rope_w224_d6_pointer_swiglu_rmsnorm_gate_conv 2400 0.001

# Stage 2: 4-fold out-of-fold teachers (each never sees its held-out quarter of the training tokens)
for k in 0 1 2 3; do
  train oof4_T384do30_f$k teacher_w384_d8_h4_do30 12000 0.001  --folds 4 --holdout-fold $k
  train oof4_S240do10_f$k rope_w240_d6_h2_do10    12000 0.0012 --folds 4 --holdout-fold $k
done

# Stage 3: first-generation student distilled from a uniform 5-teacher ensemble (it becomes a teacher itself)
train kd_ens5_w240_do05_s40000_a07 rope_w240_d6_h2_do05 40000 0.0012 --distill-alpha 0.7 --teacher $(ck \
  teacher_w384_d8_h4_do30_s12000_ema099_lr1e3 teacher_w384_d8_h4_do20_s8000_ema099_lr1e3 \
  teacher_w512_d8_h4_do40_s12000_ema099_lr1e3 teacher_w384_d8_h4_do40_s16000_ema099_lr1e3 \
  rope_w240_d6_h2_do20_s30000_ema099_lr12e4)

# Stage 4: final student. Target = 0.7 x (8-model ensemble, EM weights fitted on validation)
#                                + 0.3 x (out-of-fold ensemble, configs/oof4_T384do30_S240do10.json)
train kd_hyb_ens8w_oof4_do05_s40000_a07_m03 rope_w240_d6_h2_do05 40000 0.0012 --distill-alpha 0.7 \
  --oof-teachers configs/oof4_T384do30_S240do10.json --oof-mix 0.3 \
  --teacher $(ck kd_ens5_w240_do05_s40000_a07 teacher_w384_d8_h4_do20_s20000_ema099_lr1e3 \
    teacher_w512_d8_h4_do30_s20000_ema099_lr1e3 rope_w256_d6_h2_s2400_ema099 \
    rope_w240_d6_h2_do10_s12000_ema099_lr12e4 rope_w224_d6_gate_conv_s2400_ema099 \
    teacher_w384_d8_h4_do20_s8000_ema099_lr1e3 rope_w240_d6_h2_do20_s30000_ema099_lr12e4) \
  --teacher-weights 0.2278 0.1952 0.1972 0.0628 0.0907 0.0598 0.0914 0.0752

# Official score: full test split, CPU, FP32
python evaluate.py --checkpoint runs/kd_hyb_ens8w_oof4_do05_s40000_a07_m03/checkpoint.pt \
  --device cpu --precision fp32 --threads 4 --split test
