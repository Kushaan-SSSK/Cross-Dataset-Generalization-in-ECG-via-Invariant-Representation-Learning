#!/usr/bin/env bash
set -euo pipefail

cd /workspace/Cross-Dataset-Generalization-in-ECG-via-Invariant-Representation-Learning
source .venv/bin/activate

mkdir -p outputs/baselines
LOGDIR="outputs/baselines/_logs"
mkdir -p "$LOGDIR"

run_one () {
  name="$1"; shift
  echo "=============================="
  echo "START: $name  $(date)"
  echo "CMD: python -u $*"
  echo "LOG: $LOGDIR/${name}.log"
  echo "=============================="
  python -u "$@" 2>&1 | tee "$LOGDIR/${name}.log"
  echo "=============================="
  echo "DONE:  $name  $(date)"
  echo "=============================="
}

run_one erm_fixed  -m src.train method=erm          train.epochs=50 hydra.job.chdir=True hydra.run.dir=outputs/baselines/erm_fixed
run_one dann_fixed -m src.train method=dann         train.epochs=50 hydra.job.chdir=True hydra.run.dir=outputs/baselines/dann_fixed
run_one vrex_fixed -m src.train method=vrex         train.epochs=50 hydra.job.chdir=True hydra.run.dir=outputs/baselines/vrex_fixed
run_one irm_fixed  -m src.train method=irm          train.epochs=50 hydra.job.chdir=True hydra.run.dir=outputs/baselines/irm_fixed
run_one pid_v2     -m src.train method=disentangled train.epochs=50 hydra.job.chdir=True hydra.run.dir=outputs/baselines/v2

echo "ALL DONE ✅  $(date)"
