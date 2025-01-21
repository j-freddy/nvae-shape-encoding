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
projected_channels_list=("4")
warmup_steps_list=("6420")
seeds=("1969 1970 1971 1972 1973 1974 1975 1976 1977 1978")
logdir="logs-cnvae-no-kl"

# Train
for projected_channels in $projected_channels_list
do
    for warmup_steps in $warmup_steps_list
    do
        for seed in $seeds
        do
            model_name="pc-${projected_channels}-ws-${warmup_steps}-seed-${seed}"
            cbetas_str="2,2,2"
            betas_str="10,10,10"
            # Train
            python -m arch.cnvae.train \
                --epochs 100 \
                --arch "default" \
                --projected_channels $projected_channels \
                --warmup_steps $warmup_steps \
                --cbetas $cbetas_str \
                --betas $betas_str \
                --seed $seed \
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
        for seed in $seeds
        do
            model_name="pc-${projected_channels}-ws-${warmup_steps}-seed-${seed}"
            # Get saved model path
            model_path=$(ls ${logdir}/nvae_seg_acdc/${model_name}/checkpoints/*.ckpt)
            # Test: Save figures and metrics
            python -m arch.nvaeseg.test --model_path $model_path --logs $logdir
        done
    done
done
