# Nouveau-VAE for Anatomical Shape Encoding

<!-- TODO -->
<!-- This is a template README.md. Write this again at the very end. -->

## Usage Guide

### Installation

1. Clone this repository.

```sh
git clone https://github.com/j-freddy/nvae-shape-encoding.git
```

2. Create virtual environment with Python 3.11.8.

```sh
# Go inside repo
cd nvae-shape-encoding
# Check Python 3.11.8 is being used
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

To check everything has been set up correctly, run the commands below.
```sh
# View data samples
python -m data_viewer --dataset acdc
# Train
python -m arch.vae.train --epochs 50
python -m arch.nvae.train
# Test
python -m arch.vae.test --model_path logs/vae_acdc/version_1/checkpoints/epoch=44-step=5220.ckpt
```

### TensorBoard

```sh
tensorboard --logdir logs/vae_acdc
```

<!-- TODO -->

<!-- ## Credits -->
