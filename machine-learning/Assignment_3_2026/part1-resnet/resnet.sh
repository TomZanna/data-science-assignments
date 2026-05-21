#!/bin/bash
#SBATCH --job-name=resnet
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.err
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=120:00
#SBATCH --partition=gpu_mig

BASE_PATH=".."
. "$BASE_PATH/setup.sh"

python3 <<EOF
import papermill as pm

# lr, wd, dropout, balancing, augmentation, unfreezing
experiments = [
    # baseline
    (1e-4, 0,    0, None, "",     False, None),
    # lr tuning
    (1e-3, 0,    0, None, "",     False, None),
    (5e-5, 0,    0, None, "",     False, None),
    
    # data rebal + data aug
    (5e-5, 0,    0, None, "wcel", False,  None),
    (5e-5, 0,    0, None, "wcel", True,   None),

    (5e-5, 0,    0, None, "wrs",  True,  None),
    (5e-5, 0,    0, None, "wrs",  True,  None),

    # data unfreeze
    (5e-5, 1e-5, 0, None, "wrs",  True,  "layer4"),
    (5e-5, 5e-5, 0, None, "wrs",  True,  "layer4"),
    (5e-5, 9e-5, 0, None, "wrs",  True,  "layer4"),
]

for lr, lr4, wd, drop, bal, aug, lay in experiments:
    parts = ["resnet", f"lr{lr:.0e}", f"lr{lr4:.0e}", f"wd{wd:.0e}", bal]
    if aug: parts.append("aug")
    if lay: parts.append(f"ul{len(lay)}")
    if drop: parts.append(f"p{drop}")
    out = "-".join(parts)
    
    print(f"Running: {out}")
    
    pm.execute_notebook(
        'resnet.ipynb', f"{out}.ipynb",
        parameters={
            "weight_decay": wd,
            "lr": lr,
            "lr4": lr4,
            "class_balancing": bal, 
            "data_augmentation": aug, 
            "layers_to_unfreeze": lay, 
            "dropout": drop,
            "nb_name": out,
            "disable_tqdm": True,
        }
    )
EOF