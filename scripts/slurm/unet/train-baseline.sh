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

# ==============================================================================
# [U-Net Train]
# Train baseline model with different seeds.
# ==============================================================================

seeds=("1970 1971 1972 1973 1974")

logdir="logs-unet-baseline"

# Train

for seed in $seeds
do
    model_name="baseline-seed-${seed}"
    python -m arch.unet.train \
        --epochs 100 \
        --loss_reg "cross_entropy" \
        --augment \
        --seed $seed \
        --model_name $model_name \
        --logs $logdir
done

# Evaluate

for seed in $seeds
do
    model_name="baseline-seed-${seed}"
    # Get saved model path
    model_path=$(ls ${logdir}/unet_acdc/${model_name}/checkpoints/*.ckpt)
    # Test: Save figures and metrics
    python -m arch.unet.test --model_path $model_path --logs $logdir
done
