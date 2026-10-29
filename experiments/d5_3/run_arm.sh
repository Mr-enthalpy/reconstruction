#!/usr/bin/env bash
# D5.3 arm launcher. Every path is an explicit argument; nothing is hard-coded.
#
#   run_arm.sh <prepared> <arm> <gpu> <kernel-file> <output-dir> <seed> [config]
#
# The script caps CPU thread pools because containers often report all host CPUs
# while the lease grants only a fraction of them.
set -euo pipefail

if [ "$#" -lt 6 ]; then
  echo "usage: $0 <prepared> <arm> <gpu> <kernel-file> <output-dir> <seed> [config]" >&2
  exit 2
fi

PREPARED=$1
ARM=$2
GPU=$3
KERNEL=$4
OUT=$5
SEED=$6
CONFIG=${7:-experiments/d5_3/configs/d5_3_a6.json}

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH="$PWD/src"

echo "START $(date -u '+%FT%TZ') arm=$ARM gpu=$GPU seed=$SEED kernel=$KERNEL output=$OUT"
python3 -X utf8 experiments/d5/train.py \
  --prepared "$PREPARED" \
  --output "$OUT" \
  --config "$CONFIG" \
  --kernel-file "$KERNEL" \
  --seed "$SEED" \
  --device cuda
echo "END $(date -u '+%FT%TZ') arm=$ARM"
