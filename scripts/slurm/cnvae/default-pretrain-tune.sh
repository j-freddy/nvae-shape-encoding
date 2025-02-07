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

# Grid size is 36
projected_channels_list=("4")
# Size=1 (6420 is 214*30 so first 30 epochs)
warmup_steps_list=("6420")
# Size=6
betas=("1 3 5 7 9 11")
# Size=6
cbetas=("0 0.5 1 1.5 2 2.5")

logdir="logs-cnvae-pretrain-tune"

# Train
for projected_channels in $projected_channels_list
do
    for warmup_steps in $warmup_steps_list
    do
        for beta in $betas
        do
            for cbeta in $cbetas
            do
                model_name="pc-${projected_channels}-ws-${warmup_steps}-b-${beta}-cb-${cbeta}"
                cbetas_str="${cbeta},${cbeta},${cbeta}"
                betas_str="${beta},${beta},${beta}"
                # Train
                python -m arch.cnvae.train \
                    --epochs 100 \
                    --arch "default" \
                    --projected_channels $projected_channels \
                    --warmup_steps $warmup_steps \
                    --cbetas $cbetas_str \
                    --betas $betas_str \
                    --pretrained_nvaeseg_model_path "logs/nvae_seg_acdc/default/checkpoints/epoch=60-step=13054.ckpt" \
                    --no_warmup \
                    --model_name $model_name \
                    --logs $logdir
            done
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
            for cbeta in $cbetas
            do
                model_name="pc-${projected_channels}-ws-${warmup_steps}-b-${beta}-cb-${cbeta}"
                # Get saved model path
                model_path=$(ls ${logdir}/cnvae_acdc/${model_name}/checkpoints/*.ckpt)
                # Test: Save figures and metrics
                python -m arch.cnvae.test --model_path $model_path --logs $logdir
            done
        done
    done
done
