
# Run Baselines Script
# Usage: ./scripts/run_baselines.ps1

echo "Starting Baseline: ERM (Standard)"
python -m src.train method=erm train.epochs=50 hydra.run.dir=outputs/baselines/erm
echo "Finished ERM."

echo "Starting Baseline: DANN (Robust)"
python -m src.train method=dann train.epochs=50 hydra.run.dir=outputs/baselines/dann
echo "Finished DANN."

echo "Starting Baseline: V-REx (Robust)"
python -m src.train method=vrex train.epochs=50 hydra.run.dir=outputs/baselines/vrex
echo "Finished V-REx."

echo "All Baselines Complete."
