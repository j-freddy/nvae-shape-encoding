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

alphas=("0 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1 5 10 50 100 500 1000")

logdir="logs-unet-shape-prior"

# Train

for alpha in $alphas
do
    model_name="shape-prior-alpha-${alpha}"
    python -m arch.unet.train \
        --epochs 100 \
        --loss_reg "shape_prior" \
        --augment \
        --model_name $model_name \
        --alpha $alpha \
        --logs $logdir
done

# Evaluate

for alpha in $alphas
do
    model_name="shape-prior-alpha-${alpha}"
    # Get saved model path
    model_path=$(ls ${logdir}/unet_acdc/${model_name}/checkpoints/*.ckpt)
    # Test: Save figures and metrics
    python -m arch.unet.test --model_path $model_path --logs $logdir
done
