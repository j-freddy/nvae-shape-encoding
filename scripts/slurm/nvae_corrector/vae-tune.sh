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

betas=("0.1 0.2 0.5 1 2 5 10")
latent_dims=("8 16 32 64")

logdir="logs-vae-corrector-tune-2"

# Train
for latent_dim in $latent_dims
do
    for beta in $betas
    do
        model_name="b-${beta}-latent-${latent_dim}"
        # Train
        python -m arch.vae_corrector.train \
            --epochs 50 \
            --latent_dim $latent_dim \
            --beta $beta \
            --model_name $model_name \
            --logs $logdir
    done
done

# Evaluate
for latent_dim in $latent_dims
do
    for beta in $betas
    do
        model_name="b-${beta}-latent-${latent_dim}"
        # Get saved model path
        model_path=$(ls ${logdir}/vae_corrector_acdc/${model_name}/checkpoints/*.ckpt)
        # Test: Save figures and metrics
        python -m arch.vae_corrector.test --model_path $model_path --logs $logdir
    done
done
