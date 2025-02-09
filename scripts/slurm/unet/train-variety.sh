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

# ==============================================================================
# [Train Variety]
# Train a variety of segmentation models, then save the predicted segmentations
# to form a new dataset.
# ==============================================================================

# logdir="logs-segmentation-models"
logdir="logs-resunet"

# model_types=("unet swinunet attentionunet resunet")
model_types=("resunet")

# Train

for model_type in $model_types
do
    python -m arch.unet.train \
        --epochs 100 \
        --model_type $model_type \
        --augment \
        --model_name $model_type \
        --logs $logdir
done

# Evaluate

for model_type in $model_types
do
    # Get saved model path
    model_path=$(ls ${logdir}/unet_acdc/${model_type}/checkpoints/*.ckpt)
    # Test: Save figures and metrics
    python -m arch.unet.test --model_path $model_path --logs $logdir
done

# Save predicted segmentations

for model_type in $model_types
do
    # Get saved model path
    model_path=$(ls ${logdir}/unet_acdc/${model_type}/checkpoints/*.ckpt)
    # Save predicted segmentations
    python -m arch.unet.save_segmentations --model_path $model_path --split train --model_type $model_type
    python -m arch.unet.save_segmentations --model_path $model_path --split val --model_type $model_type
    python -m arch.unet.save_segmentations --model_path $model_path --split test --model_type $model_type
done
