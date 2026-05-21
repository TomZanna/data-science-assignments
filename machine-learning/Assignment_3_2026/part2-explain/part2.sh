#!/bin/bash
#SBATCH --job-name=part2
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.err
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00
#SBATCH --partition=gpu_mig

BASE_PATH=".."
. "$BASE_PATH/setup.sh"

jupyter-nbconvert --to notebook --execute part2.ipynb