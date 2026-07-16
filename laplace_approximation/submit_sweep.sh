#!/bin/bash
#SBATCH --job-name=LAP_M2_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH -o lap_sweep_%j.out
#SBATCH -e lap_sweep_%j.err
 
module load anaconda
conda activate your_env   # replace with your environment name
 
python LAP_M2_sweep.py \
    --m2_values 3 5 10 20 \
    --data_path /path/to/massdamper_data.pkl \
    --output sweep_results.pkl
 
echo "Done"