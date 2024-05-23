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
# NVAE ACDC: See previous tuning scripts for other configurations.
# z_channels=4
#
# Time taken: unknown
# ==============================================================================

# grid size is 54
# size=1
projected_channels_list=("16")
# size=2 (214*15 and 214*25)
warmup_steps_list=("3210 5350")
# size=3
betas0=("1000000 2500000 5000000")
# size=3
betas1=("1000000 2500000 5000000")
# size=3
betas2=("1000000 2500000 5000000")

# Uncomment for testing
# warmup_steps_list=("500")
# betas0=("1")
# betas1=("10")
# betas2=("200")

logdir="logs-nvae-3"

# Train

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
                    # Train
                    python -m arch.nvae.train \
                        --epochs 100 \
                        --projected_channels $projected_channels \
                        --z_channels 4 \
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
