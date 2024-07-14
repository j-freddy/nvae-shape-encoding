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
# [NVAE Tune Slim]
# NVAE ACDC: It seems that beta0=beta1=beta2 works well. This is a smaller grid
# where this constraint is met. For the five latent architecture.
#
# Time taken: unknown
# ==============================================================================

# Grid size is 20
projected_channels_list=("4")
# Size=1 (6420 is 214*30 so first 30 epochs)
warmup_steps_list=("6420")
# Size=6
betas=("1 2 5 10 15 20")

logdir="logs-nvae-five-latent-sr"

# Train

for projected_channels in $projected_channels_list
do
    for warmup_steps in $warmup_steps_list
    do
        for beta in $betas
        do
            model_name="pc-${projected_channels}-ws-${warmup_steps}-b-${beta}"
            betas_str="${beta},${beta},${beta},${beta},${beta}"
            # Train
            python -m arch.nvae.train \
                --epochs 100 \
                --arch "five-latent" \
                --projected_channels $projected_channels \
                --warmup_steps $warmup_steps \
                --betas $betas_str \
                --sr \
                --model_name $model_name \
                --logs $logdir
        done
    done
done

# Evaluate

for projected_channels in $projected_channels_list
do
    for warmup_steps in $warmup_steps_list
    do
        for beta in $betas
        do
            model_name="pc-${projected_channels}-ws-${warmup_steps}-b-${beta}"
            # Get saved model path
            model_path=$(ls ${logdir}/nvae_acdc/${model_name}/checkpoints/*.ckpt)
            # Test: Save figures and metrics
            python -m arch.nvae.test --model_path $model_path --logs $logdir
        done
    done
done
