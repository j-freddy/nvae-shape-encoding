#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=ffj20 # required to send email notifcations - please replace <your_username> with your college login name or email address

. /vol/cuda/12.2.0/setup.sh
export LD_LIBRARY_PATH=/vol/bitbucket/${USER}/nvae-shape-encoding/venv/lib/python3.10/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH}
TERM=vt100                # or TERM=xterm
/usr/bin/nvidia-smi
uptime

cd nvae-shape-encoding
export PATH=/vol/bitbucket/${USER}/nvae-shape-encoding/venv/bin/:$PATH
source activate

logdir="logs-nvae-best"

# Train

beta="10"
model_name="b-${beta}"
betas_str="${beta},${beta},${beta}"

python -m arch.nvae.train \
    --epochs 100 \
    --arch "default" \
    --projected_channels 4 \
    --warmup_steps 6420 \
    --betas $betas_str \
    --model_name $model_name \
    --logs $logdir

# Evaluate

# Get saved model path
model_path=$(ls ${logdir}/nvae_acdc/${model_name}/checkpoints/*.ckpt)
# Test: Save figures and metrics
python -m arch.nvae.test --model_path $model_path --logs $logdir
