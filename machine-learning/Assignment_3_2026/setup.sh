module purge
module load 2025
module load CUDA/12.9.1
module load Python/3.13.5-GCCcore-14.3.0

BASE_PATH="${BASE_PATH:-.}"
VENV_NAME="$BASE_PATH/.venv"

if [ ! -d "$VENV_NAME" ]; then
    echo "Creating venv"
    python3.13 -m venv --system-site-packages $VENV_NAME
    
    source $VENV_NAME/bin/activate
    echo "Installing requirements.txt..."
    pip install --upgrade pip
    pip install -r "$BASE_PATH/requirements.txt"
    pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp313-cp313-linux_x86_64.whl
else
    echo "Found venv"
    source $VENV_NAME/bin/activate
fi