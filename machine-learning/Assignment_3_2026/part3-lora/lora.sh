#!/bin/bash
#SBATCH --job-name=lora
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.err
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=360:00
#SBATCH --partition=gpu_h100

BASE_PATH=".."
. "$BASE_PATH/setup.sh"

python3 <<EOF
import papermill as pm

experiments = [
    (5e-6, 4, 8,  0.01, True),
    (8e-6, 4, 8,  0.01, True),
    (5e-6, 4, 8,  0.01, False),
    (5e-6, 4, 8,  0.2, True),
    (5e-6, 4, 8,  0.2, False),
    (5e-6, 4, 8,  0.1, True),
    (5e-6, 4, 8,  0.1, False),

    (1e-5, 8, 16,  0.01, True),
    (1e-5, 8, 16,  0.01, False),
    (1e-5, 16, 32,  0.01, True),
    (1e-5, 16, 32,  0.01, False),
    (1e-5, 8, 16,  0.2, True),
    (1e-5, 8, 16,  0.2, False),
    (1e-5, 16, 32,  0.2, True),
    (1e-5, 16, 32,  0.2, False),
    (1e-5, 8, 16,  0.1, True),
    (1e-5, 8, 16,  0.1, False),
    (1e-5, 16, 32,  0.1, True),
    (1e-5, 16, 32,  0.1, False),
]

for lr, r, a, dropout, all_modules in experiments:
    parts = ["lora", f"lr{lr}", f"r{r}", f"a{a}", f"d{dropout}"]
    if all_modules: parts.append("full")
    out = "-".join(parts)
    
    print(f"Running: {out}")
    
    pm.execute_notebook(
        'lora.ipynb', f"{out}.ipynb",
        parameters={
            "r_val": r,
            "alpha_val": a,
            "dropout": dropout,
            "all_modules": all_modules,
            "nb_name": out,
            "lr": lr,
            "disable_tqdm": True,
        }
    )
EOF