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
# NVAE: Train the 3 best configurations
#
# This is a lightweight script that summarises the results from all
# previous hyperparameter tuning.
# ==============================================================================

logdir="logs-nvae-best"

# Expect: 0.961 DSC, 35.27 FRDS, 0.9749 valid

python -m arch.nvae.train \
    --epochs 100 \
    --arch "default" \
    --projected_channels 4 \
    --warmup_steps 6420 \
    --betas "10,9,11" \
    --model_name "default" \
    --logs $logdir

# Expect: 0.987 DSC, 37.4 FRDS, 0.859 valid

python -m arch.nvae.train \
    --epochs 100 \
    --arch "default" \
    --projected_channels 4 \
    --min_channels 16 \
    --warmup_steps 6420 \
    --betas "1,1,1" \
    --sr \
    --model_name "default-clamp-sr" \
    --logs $logdir

# Expect: 0.998 DSC, 83.1 FRDS, 0.721 valid

python -m arch.nvae.train \
    --epochs 100 \
    --arch "latent-skip" \
    --projected_channels 4 \
    --warmup_steps 6420 \
    --betas "1,1,1" \
    --model_name "latent-skip" \
    --logs $logdir
