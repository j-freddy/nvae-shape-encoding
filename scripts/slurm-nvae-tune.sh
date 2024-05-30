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
# [NVAE Tune]
# NVAE ACDC: Grid search on hyperparameters: project channel size, warmup steps,
# beta values. With z_channels=20.
#
# Note: A larger range of hyperparameters has been previously tuned (e.g. 500,
# 3210 warmup steps and much lower beta values). This is the most recent grid
# search conditioned on previous optimal results.
#
# Time taken: unknown
# ==============================================================================

echo "Starting NVAE tune..."

# grid size is 54
# size=2
projected_channels_list=("8 16")
# size=1 (5350 is 214*25 so first 25 epochs)
warmup_steps_list=("5350")
# size=3
betas0=("500000 1500000 5000000")
# size=3
betas1=("250000 750000 2500000")
# size=3
beta2=("500000 1500000 5000000")

logdir="logs-nvae-latent-20"

echo "Grid search set up."

# Train

for projected_channels in $projected_channels_list
do
    echo "Loop 1"
    for warmup_steps in $warmup_steps_list
    do
        echo "Loop 2"
        for beta0 in $betas0
        do
            echo "Loop 3"
            for beta1 in $betas1
            do
                echo "Loop 4"
                for beta2 in $betas2
                do
                    model_name="pc-${projected_channels}-ws-${warmup_steps}-b0-${beta0}-b1-${beta1}-b2-${beta2}"
                    echo "Training model: $model_name"
                    # Train
                    python -m arch.nvae.train \
                        --epochs 100 \
                        --projected_channels $projected_channels \
                        --z_channels 20 \
                        --warmup_steps $warmup_steps \
                        --beta0 $beta0 \
                        --beta1 $beta1 \
                        --beta2 $beta2 \
                        --model_name $model_name \
                        --logs $logdir
                done
            done
        done
    done
done

# Evaluate

for projected_channels in $projected_channels_list
do
    for warmup_steps in $warmup_steps_list
    do
        for beta0 in $betas0
        do
            for beta1 in $betas1
            do
                for beta2 in $betas2
                do
                    model_name="pc-${projected_channels}-ws-${warmup_steps}-b0-${beta0}-b1-${beta1}-b2-${beta2}"
                    # Get saved model path
                    model_path=$(ls ${logdir}/nvae_acdc/${model_name}/checkpoints/*.ckpt)
                    # Test: Save figures and metrics
                    python -m arch.nvae.test --model_path $model_path --logs $logdir
                done
            done
        done
    done
done
