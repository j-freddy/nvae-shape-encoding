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

centres=("1 2 3 4 5")
seeds=("1970" "1971" "1972" "1973" "1974")

logdir="logs-mnms-no-pretrain"

# Train

for centre in $centres
do
    for i in {0..4}
    do
        seed=${seeds[i]}
        
        # Baseline
        model_name="baseline-centre-${centre}-seed-${seed}"

        python -m arch.unet.train \
            --epochs 50 \
            --loss_reg "cross_entropy" \
            --dataset mnms \
            --centre $centre \
            --num_subjects 5 \
            --sort_by_validity \
            --augment \
            --seed $seed \
            --model_name $model_name \
            --logs $logdir

        # Shape prior
        model_name="shape-prior-centre-${centre}-seed-${seed}"

        python -m arch.unet.train \
            --epochs 50 \
            --loss_reg "shape_prior" \
            --dataset mnms \
            --centre $centre \
            --num_subjects 5 \
            --sort_by_validity \
            --augment \
            --seed $seed \
            --model_name $model_name \
            --logs $logdir
    done
done

# Evaluate

for centre in $centres
do
    for i in {0..4}
    do
        seed=${seeds[i]}

        # Baseline
        model_name="baseline-centre-${centre}-seed-${seed}"
        # Get saved model path
        model_path=$(ls ${logdir}/unet_mnms/${model_name}/checkpoints/*.ckpt)

        python -m arch.unet.test \
            --model_path $model_path \
            --dataset mnms \
            --centre $centre \
            --logs $logdir

        # Shape prior
        model_name="shape-prior-centre-${centre}-seed-${seed}"
        # Get saved model path
        model_path=$(ls ${logdir}/unet_mnms/${model_name}/checkpoints/*.ckpt)

        python -m arch.unet.test \
            --model_path $model_path \
            --dataset mnms \
            --centre $centre \
            --logs $logdir
    done
done
