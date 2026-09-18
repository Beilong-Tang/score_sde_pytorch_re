#!/usr/bin/env bash
# Train on 4 GPUs (via torchrun), then sample 5k images and score them
# (Inception score / FID / KID) against precomputed CIFAR-10 stats.
#
# Usage:
#   ./train_and_eval_4gpu.sh
#
# Edit the variables below to change the config, workdir, GPU count, or
# number of eval samples.
set -euo pipefail

CONFIG="configs/vp/cifar10_ddpmpp_continuous.py"
WORKDIR="workdir/cifar10_ddpmpp_continuous"
NPROC_PER_NODE=4
NUM_SAMPLES=50000
EVAL_BATCH_SIZE=1024

cd "$(dirname "$0")"

# FID/KID need reference Inception activations for the training set.
if [ ! -f "assets/stats/cifar10_stats.npz" ]; then
  echo "==> Precomputing CIFAR-10 Inception stats..."
  python compute_dataset_stats.py
fi

echo "==> Training on ${NPROC_PER_NODE} GPUs..."
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" main.py \
  --config="${CONFIG}" \
  --mode=train \
  --workdir="${WORKDIR}"

# Evaluation (run_lib.evaluate) is single-process only, so no torchrun here.
# Pick the checkpoint with the highest index written by training.
LATEST_CKPT=$(ls "${WORKDIR}/checkpoints" | grep -oE '[0-9]+' | sort -n | tail -1)
if [ -z "${LATEST_CKPT}" ]; then
  echo "No checkpoints found in ${WORKDIR}/checkpoints" >&2
  exit 1
fi
echo "==> Sampling ${NUM_SAMPLES} images and evaluating checkpoint_${LATEST_CKPT}..."

python main.py \
  --config="${CONFIG}" \
  --mode=eval \
  --workdir="${WORKDIR}" \
  --eval_folder="eval_${NUM_SAMPLES}" \
  --config.eval.enable_sampling=True \
  --config.eval.enable_loss=False \
  --config.eval.enable_bpd=False \
  --config.eval.num_samples="${NUM_SAMPLES}" \
  --config.eval.batch_size="${EVAL_BATCH_SIZE}" \
  --config.eval.begin_ckpt="${LATEST_CKPT}" \
  --config.eval.end_ckpt="${LATEST_CKPT}"

echo "==> Done. Results in ${WORKDIR}/eval_${NUM_SAMPLES}/report_${LATEST_CKPT}.npz"
