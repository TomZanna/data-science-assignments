#!/bin/bash
#SBATCH --job-name=rag
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.err
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=120:00
#SBATCH --partition=gpu_a100

BASE_PATH=".."
. "$BASE_PATH/setup.sh"
 
python3 <<EOF
import papermill as pm

experiments = [
    (3,  256, 20 ),
    (3,  256, 50 ),
    (3,  512, 50 ),
    (3,  512, 100),
    (3, 1024, 50 ),
    (3, 1024, 100),
    (3, 1024, 200),
    
    (6,  256, 20 ),
    (6,  256, 50 ),
    (6,  512, 50 ),
    (6,  512, 100),
    (6, 1024, 50 ),
    (6, 1024, 100),
    (6, 1024, 200),

    (12,  256, 20 ),
    (12,  256, 50 ),
    (12,  512, 50 ),
    (12,  512, 100),
    (12, 1024, 50 ),
    (12, 1024, 100),
    (12, 1024, 200),
]

for top_k, s, o in experiments:
    parts = ["rag", f"tk{top_k}", f"s{s}", f"o{o}"]
    out = "-".join(parts)
    
    print(f"Running: {out}")
    
    pm.execute_notebook(
        'rag.ipynb', f"{out}.ipynb",
        parameters={
            "chunk_overlap": o,
            "chunk_size": s,
            "top_k_documents": top_k,
            "docs_dir": "../docs",
        },
    )
EOF
