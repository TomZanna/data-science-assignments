#!/bin/bash
#SBATCH --job-name=vit
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.err
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=240:00
#SBATCH --partition=gpu_mig

BASE_PATH=".."
. "$BASE_PATH/setup.sh"

python3 <<EOF
import papermill as pm

# lr, wd, dropout, balancing, augmentation
experiments = [
    # baseline
    (3e-4, 0, None, "", False),
    # lr tuning
    (1e-3, 0, None, "", False),
    (1e-5, 0, None, "", False),

    # data augmentation
    (3e-4, 0, None, "wcel", False),
    (3e-4, 0, None, "wcel", True),

    (3e-4, 0, None, "wrs", False),
    (3e-4, 0, None, "wrs", True),

    # tuning
    (3e-4, 0, 0.2, "wrs", True),
    (3e-4, 0, 0.35, "wrs", True),
    (3e-4, 0, 0.5, "wrs", True),
]

for lr, wd, drop, bal, aug in experiments:
    parts = ["vit", f"lr{lr:.0e}", f"wd{wd:.0e}", bal]
    if aug: parts.append("aug")
    if drop: parts.append(f"p{drop}")
    out = "-".join(parts)
    
    print(f"Running: {out}")
    
    pm.execute_notebook(
        'vit.ipynb', f"{out}.ipynb",
        parameters={
            "weight_decay": wd,
            "lr": lr,
            "class_balancing": bal, 
            "data_augmentation": aug, 
            "dropout": drop,
            "nb_name": out,
            "disable_tqdm": True,
        }
    )
EOF
