# Nouveau-VAE for Anatomical Shape Encoding

<!-- TODO -->
<!-- This is a template README.md. Write this again at the very end. -->

## Usage Guide

### Quick Start

1. Clone this repository.

```sh
git clone https://github.com/j-freddy/nvae-shape-encoding.git
```

2. Create virtual environment with Python 3.10+. Python 3.11.8 is recommended.

```sh
# Go inside repo
cd nvae-shape-encoding
# Check Python 3.10+ is being used
python --version
# Create virtual environment
python -m venv venv
# Activate virtual environment
source venv/bin/activate
```

3. Install dependencies.

```sh
pip install -r requirements.txt
```

If everything has been set up correctly, the commands below should work.
```sh
# View data samples
python -m data_viewer --dataset acdc
# Train a VAE model with good configurations
python -m arch.vae.train \
    --epochs 1 \
    --latent_dim 8 \
    --beta 0.1 \
    --gamma 1000 \
    --loss_reg info_vae \
    --register_alignment \
    --augment
# Train a NVAE model with good configurations
python -m arch.nvae.train --epochs 50
# Test
python -m arch.vae.test --model_path path/to/vae/checkpoint.ckpt --register_alignment
python -m arch.nvae.test --model_path path/to/nvae/checkpoint.ckpt

# A typical checkpoint path is:
# logs/vae_acdc/version_0/checkpoints/epoch=35-step=7704.ckpt
```

### TensorBoard

```sh
tensorboard --logdir logs/vae_acdc
tensorboard --logdir logs/nvae_acdc
```

<!-- TODO -->

<!-- ## Credits -->
