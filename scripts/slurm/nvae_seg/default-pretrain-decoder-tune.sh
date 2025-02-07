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

# Grid size is 22
projected_channels_list=("4")
# Size=1 (6420 is 214*30 so first 30 epochs)
warmup_steps_list=("6420")
# Size=22
betas=("0 0.01 0.02 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 2 3 4 5 6 7 8 9 10")

logdir="logs-nvaeseg-pretrain-decoder-tune"

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
            python -m arch.nvaeseg.train \
                --epochs 100 \
                --arch "default" \
                --projected_channels $projected_channels \
                --warmup_steps $warmup_steps \
                --betas $betas_str \
                --model_name $model_name \
                --logs $logdir \
                --pretrained_nvae_model_path "logs/nvae_acdc/best/default/checkpoints/epoch=97-step=20972.ckpt"
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
            model_path=$(ls ${logdir}/nvae_seg_acdc/${model_name}/checkpoints/*.ckpt)
            # Test: Save figures and metrics
            python -m arch.nvaeseg.test --model_path $model_path --logs $logdir
        done
    done
done
