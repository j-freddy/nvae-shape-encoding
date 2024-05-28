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
# [InfoVAE Tune with Augmentation]
# VAE ACDC: Grid search on beta (KL), gamma (KL[q(z) || p(z)]) and latent dim
# hyperparameters.
#
# Time taken: unknown
# ==============================================================================

# grid size is 120
# size=5
latent_dims=("2 4 6 8 16")
# size=2
betas=("0 1")
# size=12
gammas=("1 2 5 10 20 50 100 200 500 1000 2000 5000")

logdir="logs-info-vae-augment"

# Train

for latent_dim in $latent_dims
do
    for beta in $betas
    do
        for gamma in $gammas
        do
            model_name="ld-${latent_dim}-beta-${beta}-gamma-${gamma}"
            # Train
            python -m arch.vae.train \
                --epochs 50 \
                --latent_dim $latent_dim \
                --beta $beta \
                --gamma $gamma \
                --model_name $model_name \
                --loss_reg info_vae \
                --augment \
                --logs $logdir
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
            model_path=$(ls ${logdir}/vae_acdc/${model_name}/checkpoints/*.ckpt)
            # Test: Save figures and metrics
            python -m arch.vae.test \
                --model_path $model_path \
                --augment \
                --logs $logdir
        done
    done
done
