#!/bin/bash
#SBATCH --job-name=both_180
#SBATCH --partition=AMD
#SBATCH --nodelist=pikrit
#SBATCH --cpus-per-task=1
#SBATCH --time=5-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

export PYTHONUNBUFFERED=1

echo "[DEBUG] Job started on $(hostname) at $(date)"
cd ~/Code\ QuGAN\ 7 || { echo "Directory not found"; exit 1; }
source myenv/bin/activate
echo "[DEBUG] Python version:"; python --version
echo "[DEBUG] Starting main.py..."
python main.py
echo "[DEBUG] main.py finished at $(date)"