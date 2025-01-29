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

# Grid size is 40
z_channels_list=("32 64 128 256")
# Size=1 (6420 is 214*30 so first 30 epochs)
warmup_steps_list=("6420")
# Size=10
betas=("1 1.125 1.25 1.5 2 3 4 5 7 10")

logdir="logs-nvaeseg-tune"

# Train

for z_channels in $z_channels_list
do
    for warmup_steps in $warmup_steps_list
    do
        for beta in $betas
        do
            model_name="z-${z_channels}-ws-${warmup_steps}-b-${beta}"
            betas_str="${beta},${beta},${beta}"
            # Train
            python -m arch.nvaeseg.train \
                --epochs 100 \
                --arch "default" \
                --projected_channels 4 \
                --warmup_steps $warmup_steps \
                --z_channels $z_channels \
                --betas $betas_str \
                --model_name $model_name \
                --logs $logdir
        done
    done
done

# Evaluate

for z_channels in $z_channels_list
do
    for warmup_steps in $warmup_steps_list
    do
        for beta in $betas
        do
            model_name="z-${z_channels}-ws-${warmup_steps}-b-${beta}"
            # Get saved model path
            model_path=$(ls ${logdir}/nvae_seg_acdc/${model_name}/checkpoints/*.ckpt)
            # Test: Save figures and metrics
            python -m arch.nvaeseg.test --model_path $model_path --logs $logdir
        done
    done
done
