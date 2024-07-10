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
# Any script to train a single NVAE model.
# ==============================================================================

projected_channels_list=("4")
warmup_steps_list=("6420")
betas=("1 2 5 10")

logdir="logs-nvae-dec-res"

# Train

for projected_channels in $projected_channels_list
do
    for warmup_steps in $warmup_steps_list
    do
        for beta in $betas
        do
            model_name="pc-${projected_channels}-ws-${warmup_steps}-b-${beta}"
            betas_str="${beta},${beta},${beta}"
            # Train
            python -m arch.nvae.train \
                --epochs 100 \
                --arch "default" \
                --projected_channels $projected_channels \
                --warmup_steps $warmup_steps \
                --betas $betas_str \
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
