#!/bin/bash
# --------------------
### Directives Section
# --------------------
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=ghx4
#SBATCH --time=2:20:00
#SBATCH --job-name=tiny-wiz-2
#SBATCH --account=bewi-dtai-gh
### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH -o %J.out

# Load modules first
module load python/miniforge3_pytorch/2.7.0

# Then run nvidia-smi
echo "=== GPU Info ==="
nvidia-smi

# Now check PyTorch CUDA
echo ""
echo "=== PyTorch CUDA Check ==="
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('Device count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"

pip3 install -r requirements.txt

python3 Scripts/test_linger.py