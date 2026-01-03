
# Retrain Baselines (Fixing Overwrite Issue) & Train V2
# Usage: ./scripts/retrain_baselines.ps1

echo "Starting Recovery: ERM"
python -m src.train method=erm train.epochs=50 hydra.job.chdir=True hydra.run.dir=outputs/baselines/erm_fixed
echo "Finished ERM."

echo "Starting Recovery: DANN"
python -m src.train method=dann train.epochs=50 hydra.job.chdir=True hydra.run.dir=outputs/baselines/dann_fixed
echo "Finished DANN."

echo "Starting New Training: Disentangled V2"
python -m src.train method=disentangled train.epochs=50 hydra.job.chdir=True hydra.run.dir=outputs/baselines/v2
echo "Finished V2."

echo "All Training Complete."
