#!/bin/bash
#SBATCH --job-name=dgl_split_learning
#SBATCH --partition=gpu-day-long
#SBATCH --output=dgl_%j.out
#SBATCH --error=dgl_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00



# Activate conda
source /usr/local/anaconda3/etc/profile.d/conda.sh
# Activate environment
conda activate venv

## Load modules
#module load python/3.8
#module load cuda/11.3

## Activate virtual environment
#source ~/venv/bin/activate

# Run DGL Split Learning
echo "Starting DGL Split Learning..."
python split_dgl_implementation.py

echo "Starting Baseline Split Learning..."
python baseline_split_learning.py

echo "Generating comparisons..."
python compare_results.py

echo "Training completed!"