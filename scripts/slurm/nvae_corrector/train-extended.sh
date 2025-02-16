#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=ffj20 # required to send email notifcations - please replace <your_username> with your college login name or email address

. /vol/cuda/12.2.0/setup.sh
export LD_LIBRARY_PATH=/vol/bitbucket/${USER}/nvae-shape-encoding/venv/lib/python3.12/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH}
TERM=vt100                # or TERM=xterm
/usr/bin/nvidia-smi
uptime

cd nvae-shape-encoding
export PATH=/vol/bitbucket/${USER}/nvae-shape-encoding/venv/bin/:$PATH
source activate

# Train NVAE Corrector
#
# This is phase 2: take the predicted segmentations and train a reconstruction
# model. This is an extended corrector; it takes predictions from 4 segmentation
# models. As such, it uses 6420*4=25680 warmup steps (=30 epochs).

logdir="logs-nvae-corrector"

# Train

model_name="default-extended"

python -m arch.nvae_corrector.train \
    --epochs 100 \
    --arch "default" \
    --projected_channels 4 \
    --warmup_steps 25680 \
    --betas 8,9,10 \
    --model_name $model_name \
    --logs $logdir

# Evaluate

# Get saved model path
model_path=$(ls ${logdir}/nvae_corrector_acdc/${model_name}/checkpoints/*.ckpt)
# Test: Save figures and metrics
python -m arch.nvae_corrector.test --model_path $model_path --logs $logdir
