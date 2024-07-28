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
# [U-Net Tune Augmentations]
# For the baseline U-Net, see what augmentations work best.
#
# Time taken: unknown
# ==============================================================================

gamma_ranges=("0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9")
noise_sigmas=("0 0.1 0.2 0.3 0.4 0.5")

logdir="logs-unet-baseline-tune-augs"

# Train

for gamma in $gamma_ranges
do
    for noise_sigma in $noise_sigmas
    do
        model_name="gamma-${gamma}-noise-${noise_sigma}"
        # Train
        python -m arch.unet.train \
            --epochs 50 \
            --loss_reg "cross_entropy" \
            --augment \
            --gamma_range $gamma \
            --noise_sigma $noise_sigma \
            --model_name $model_name \
            --logs $logdir
    done
done

for gamma in $gamma_ranges
do
    for noise_sigma in $noise_sigmas
    do
        model_name="gamma-${gamma}-noise-${noise_sigma}"
        # Get saved model path
        model_path=$(ls ${logdir}/unet_acdc/${model_name}/checkpoints/*.ckpt)
        # Test: Save figures and metrics
        python -m arch.unet.test --model_path $model_path --logs $logdir
    done
done
