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
# [VAE Tune]
# VAE ACDC: Grid search on beta (KL) and latent dim hyperparameters.
#
# Time estimate: 4 min per run => 7 hrs for 96 runs
# ==============================================================================

latent_dims=("2 4 8 16 32 64")
betas=("0.01 0.02 0.05 0.1 0.2 0.5 1 2 5 10 20 50 100 200 500 1000")

# Train

for latent_dim in $latent_dims
do
    for beta in $betas
    do
        model_name="ld-${latent_dim}-beta-${beta}"
        # Train
        python -m arch.vae.train --epochs 50 --latent_dim $latent_dim --beta $beta --model_name $model_name
    done
done

# Evaluate

for latent_dim in $latent_dims
do
    for beta in $betas
    do
        model_name="ld-${latent_dim}-beta-${beta}"
        # Get saved model path
        model_path=$(ls logs/vae_acdc/${model_name}/checkpoints/*.ckpt)
        # Test: Save figures and metrics
        python -m arch.vae.test --model_path $model_path --save_fig 
    done
done
