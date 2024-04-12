# Nouveau-VAE for Anatomical Shape Encoding

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
python -m train
```

<!-- TODO -->

<!-- ## Credits -->
