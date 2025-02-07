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
# [NVAE Tune]
# Note: A larger range of hyperparameters has been previously tuned
# 
# NVAE ACDC: It seems that beta=10 works well. This is a smaller grid around
# this value. For the default architecture.
#
# Time taken: 71 hr 1 min
# ==============================================================================

# Size=64
betas1=("8 9 10 11")
betas2=("8 9 10 11")
betas3=("8 9 10 11")

logdir="logs-nvae-fine"

# Train

for beta1 in $betas1
do
    for beta2 in $betas2
    do
        for beta3 in $betas3
        do
            model_name="b1-${beta1}-b2-${beta2}-b3-${beta3}"
            betas_str="${beta1},${beta2},${beta3}"
            # Train
            python -m arch.nvae.train \
                --epochs 100 \
                --arch "default" \
                --projected_channels 4 \
                --warmup_steps 6420 \
                --betas $betas_str \
                --model_name $model_name \
                --logs $logdir
        done
    done
done

# Evaluate

for beta1 in $betas1
do
    for beta2 in $betas2
    do
        for beta3 in $betas3
        do
            model_name="b1-${beta1}-b2-${beta2}-b3-${beta3}"
            # Get saved model path
            model_path=$(ls ${logdir}/nvae_acdc/${model_name}/checkpoints/*.ckpt)
            # Test: Save figures and metrics
            python -m arch.nvae.test --model_path $model_path --logs $logdir
        done
    done
done
