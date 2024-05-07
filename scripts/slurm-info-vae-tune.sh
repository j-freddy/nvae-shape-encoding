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
# [InfoVAE Tune]
# VAE ACDC: Grid search on beta (KL), gamma (KL[q(z) || p(z)]) and latent dim
# hyperparameters.
#
# Time taken: 8 hr 31 min
# ==============================================================================

# Try beta=1 at least, as a beta<1 means the expression is not guaranteed to be
# a lower bound

# Using a coarser grid search than beta-VAE due to the extra gamma
# hyperparameter

# grid size is 192
# size=4
latent_dims=("4 8 16 32")
# size=6
betas=("0 0.01 0.05 0.1 0.5 1 5")
# size=8
gammas=("1 5 10 50 100 500 1000 5000")

# Train

for latent_dim in $latent_dims
do
    for beta in $betas
    do
        for gamma in $gammas
        do
            model_name="ld-${latent_dim}-beta-${beta}-gamma-${gamma}"
            # Train
            python -m arch.vae.train --epochs 50 --latent_dim $latent_dim --beta $beta --gamma $gamma --model_name $model_name --loss_reg info_vae
        done
    done
done

# Evaluate

for latent_dim in $latent_dims
do
    for beta in $betas
    do
        for gamma in $gammas
        do
            model_name="ld-${latent_dim}-beta-${beta}-gamma-${gamma}"
            # Get saved model path
            model_path=$(ls logs/vae_acdc/${model_name}/checkpoints/*.ckpt)
            # Test: Save figures and metrics
            python -m arch.vae.test --model_path $model_path
        done
    done
done
