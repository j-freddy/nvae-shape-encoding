# Cardiac Shape Analysis with Nouveau Variational Autoencoder

Codebase for the experiments conducted for the paper "Cardiac Shape Analysis with Nouveau Variational Autoencoder", submitted as part of my dissertation for MSc degree in Computing (Artificial Intelligence and Machine Learning), Imperial College London.

Time estimations in this document are based on a 16GB RAM, 8-core CPU powered
laptop. Running with MPS (Apple silicon chip) is ~2 times faster. Running on an
A30 GPU with 24GB RAM is ~5 times faster.

To set up this repository for usage, go through [Quick Start](#quick-start) up
to (and including) the Install additional prerequisites step.

Abstract:
> Cardiovascular diseases (CVDs) cause over 20 million deaths annually, with a
third occuring prematurely in people under the age of 70. However, CVDs are
largely preventable with early detection and intervention. Over recent years,
there has been rapid progression in the development of automated techniques
for cardiac magnetic resonance imaging (MRI) analysis. Accurate delineation of
cardiac components is crucial to assist in anomaly detection and diagnosis,
and shape analysis is an essential prerequisite.<br /><br />
The emergence of deep learning has introduced powerful frameworks capable of
automating the process of learning compact shape representations. Variational
autoencoders (VAEs) are a class of generative models that excel at learning
efficient low-dimensional representations of complex data. In particular, the
Nouveau VAE (NVAE) is a deep hierarchical VAE that is the state-of-the-art among
its class in encoding fine-grained details in high-resolution images.<br /><br
/>
In this dissertation, we examine how the NVAE framework can be applied to
cardiac shape analysis. We propose configurations that can learn from clinically
annotated segmentation masks to efficiently encode cardiac anatomic shapes, with
significantly improved performance over existing VAE models (up to 0.108 Dice
increase for reconstructed masks and 22.0% anatomical validity increase in
synthetic masks when used as a generative model, the latter of which ensures the
generated shapes conform to realistic cardiac anatomy). Furthermore, we propose
a novel metric, the Fréchet ResNet Distance with SimCLR (FRDS), which improves
over the Fréchet Inception Distance in measuring the similarity between
synthetic and real cardiac segmentation masks. We demonstrate that the learned
NVAE encodings can be used in downstream tasks by using them as an anatomical
constraint to improve the segmentation performance of a U-Net model (5.3%
anatomical validity increase). We find these encodings to generalise well when
applied to unseen data, without the need for further training.

## Table of Contents

- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Trained Model Archive](#trained-model-archive)
- [Usage Guide](#usage-guide)
    - [Variational Autoencoder](#variational-autoencoder)
    - [Nouveau Variational Autoencoder](#nouveau-variational-autoencoder)
    - [U-Net](#u-net)
    - [SimCLR](#simclr)
    - [TensorBoard](#tensorboard)
- [Acknowledgements](#acknowledgements)

## Quick Start

1. Clone this repository.

```sh
git clone https://github.com/j-freddy/nvae-shape-encoding.git
```

2. Create virtual environment. The experiments are conducted with Python 3.11.8.

```sh
# Go inside repo
cd nvae-shape-encoding
# Check Python version
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

4. Install additional prerequisites.

Some scripts require pretrained models. Download the following files from [Zenodo][zenodo-model-archive]:
- `simclr_acdc.zip` - Required for VAE and NVAE testing.
- `nvae_acdc.zip` - Required for Anatomically Constrained U-Net training and
  testing.

Unzip both files and place the unzipped folders in the `logs/` subdirectory.

[zenodo-model-archive]: https://zenodo.org/uploads/13368002

If everything has been set up correctly, the commands below should work.
```sh
# View data samples
python -m utils.data_viewer --dataset acdc
# Train a baseline VAE model with good configurations (~10 minutes)
python -m arch.vae.train \
    --epochs 50 \
    --latent_dim 8 \
    --beta 0 \
    --gamma 200 \
    --loss_reg info_vae
# Test (~5 minutes)
# A typical checkpoint path is:
# logs/vae_acdc/version_0/checkpoints/epoch=45-step=4922.ckpt
python -m arch.vae.test --model_path path/to/vae/checkpoint.ckpt
```

Use [TensorBoard](#tensorboard) to see the train graphs and test metrics.

For more examples, see the respective sections:
- [Variational Autoencoder](#variational-autoencoder)
- [Nouveau Variational Autoencoder](#nouveau-variational-autoencoder)
- [U-Net](#u-net)
- [SimCLR](#simclr)
- [TensorBoard](#tensorboard)

## Repository Structure

- `analysis/` - Main evaluation metrics are calculated within the Lightning test
  step, e.g. in `arch/nvae/nvae.py`. This subdirectory contains additional
  scripts for closer inspection of the trained models. These scripts are
  presented as Jupyter notebooks to allow easy configuration and interaction.
  Each notebook contains comments and explanations. To run the notebooks, either
  replace the model path with your own trained model, or download the pretrained
  models [here][zenodo-model-archive].
- `arch/` - Implementation of frameworks. For each architecture, the train and
  test entry points are `train.py` and `test.py`.
    - `nvae/` - Nouveau-VAE (NVAE) framework.
    - `simclr/` - SimCLR framework.
    - `vae/` - Variational Autoencoder (VAE) framework.
    - `unet/` - U-Net and Anatomically Constrained U-Net frameworks.
- `data_modules/` - Lightning DataModule classes. `acdc.py` and `mnms.py` are
  used for the ACDC and M&Ms datasets, respectively. The files also contain preprocessing scripts.
- `datasets/` - Custom Torch datasets. Augmentation pipelines are implemented
  here and run during train time (if configured).
- `plot/` - Basic plots can be found in [TensorBoard](#tensorboard). This
  subdirectory contains scripts for additional plots that are not generated by
  TensorBoard, e.g. aggregate scatter plots of hyperparameter tuning results.
- `scripts/` - Shell scripts. Mostly for running hyperparameter tuning on the
  Imperial College DoC GPU cluster.
- `test/` - Unit tests for the codebase. Currently only testing the correctness
  of the `AnatomicalValidityChecker` class.
- `utils/` - Utility functions and scripts such as data viewer and scraping data
  off of TensorBoard logs.

Running programs can generate the following subdirectories.

- `data/` - Downloaded datasets and preprocessed dataset checkpoints.
- `logs/` - TensorBoard logs and model checkpoints. Also contains summary
  statistics generated from the TensorBoard scraper.

## Trained Model Archive

A collection of trained models and logs is available on
[Zenodo][zenodo-model-archive]. The performance of these models is published in
 the dissertation. To use these models, download the zip files and extract them
into the `logs/` subdirectory. Note that this collection is not a complete
archive and does not contain hyperparameter tuning experiments.

Raw link:
- https://zenodo.org/uploads/13368002

## Usage Guide

See [Quick Start](#quick-start) on setting up the repository. This section
provides detailed instructions and options on running the scripts.

All entry points should be run as modules from the root directory. For example,
use `python -m arch.nvae.train`, not `python arch/nvae/train.py`.

### Variational Autoencoder

The single-layer variational autoencoder (VAE) acts as the baseline for this
project. It takes in a previously segmented GT cardiac mask as input and outputs
a reconstruction. The code is located in `arch/vae/` and the entry points are
`train.py` and `test.py`. Frameworks include $\beta$-VAE (`vae.py`), as well as
InfoVAE implemented with minibatch sampling (InfoVAE-M / `info_vae.py`) and with
an adversarial network (InfoVAE-D / `info_adversarial_vae.py`).

#### Example

```sh
# View data samples
python -m utils.data_viewer --dataset acdc
# Train an Info-VAE model with good configurations (~10 minutes)
python -m arch.vae.train \
    --epochs 50 \
    --latent_dim 8 \
    --beta 0 \
    --gamma 200 \
    --loss_reg info_vae
# Test (~5 minutes)
# A typical checkpoint path is:
# logs/vae_acdc/version_0/checkpoints/epoch=45-step=4922.ckpt
python -m arch.vae.test --model_path path/to/vae/checkpoint.ckpt
```

Use [TensorBoard](#tensorboard) to see the train graphs and test metrics.

#### Training

```sh
python -m arch.vae.train -h

usage: train.py [-h] [--epochs EPOCHS] [--latent_dim LATENT_DIM] [--loss_reg {beta_vae,info_vae,info_adversarial_vae}]
                [--beta BETA] [--gamma GAMMA] [--filter_empty | --no-filter_empty] [--model_name MODEL_NAME] [--logs LOGS]
                [--register_alignment | --no-register_alignment] [--augment | --no-augment]

options:
  -h, --help            show this help message and exit
  --epochs EPOCHS       Max number of epochs.
  --latent_dim LATENT_DIM
                        Dimension of latent space.
  --loss_reg {beta_vae,info_vae,info_adversarial_vae}
                        Regulariser technique.
  --beta BETA           Beta value for KL divergence.
  --gamma GAMMA         Gamma value for divergence between q(z) and p(z).
  --filter_empty, --no-filter_empty
                        If set, filter out empty masks.
  --model_name MODEL_NAME
                        Directory name of saved model checkpoints and metadata.
  --logs LOGS           Root save directory for logs.
  --register_alignment, --no-register_alignment
                        If set, use masks that have been rotated such that the right ventricle points upwards.
  --augment, --no-augment
                        If set, augment training data with small random rotation.
```

#### Testing

If the model was trained with `--register_alignment` or `--augment`, the same
flag(s) must be set during testing.

```sh
python -m arch.vae.test -h

usage: test.py [-h] --model_path MODEL_PATH [--logs LOGS] [--register_alignment | --no-register_alignment]
               [--augment | --no-augment]

options:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        Path to model checkpoint.
  --logs LOGS           Root save directory for logs.
  --register_alignment, --no-register_alignment
                        If set, use masks that have been rotated such that the right ventricle points upwards.
  --augment, --no-augment
                        If set, augment training data with small random rotation.
```

See `analysis/vae` for further analysis on trained models.

### Nouveau Variational Autoencoder

Nouveau Variational Autoencoder (NVAE) is the main framework for this project.
It takes in a previously segmented GT cardiac mask as input and outputs a
reconstruction. The code is located in `arch/nvae/` and the entry points are
`train.py` and `test.py`.

#### Example

```sh
# View data samples
python -m utils.data_viewer --dataset acdc
# Train a NVAE model with good configurations (~120 minutes)
python -m arch.nvae.train \
    --epochs 100 \
    --arch "default" \
    --projected_channels 4 \
    --warmup_steps 6420 \
    --betas 10,10,10
# Test (~5 minutes)
# A typical checkpoint path is:
# logs/nvae_acdc/version_0/checkpoints/epoch=97-step=20972.ckpt
python -m arch.vae.test --model_path path/to/nvae/checkpoint.ckpt
```

Use [TensorBoard](#tensorboard) to see the train graphs and test metrics.

#### Training

```sh
python -m arch.nvae.train -h

usage: train.py [-h] [--epochs EPOCHS] [--projected_channels PROJECTED_CHANNELS] [--z_channels Z_CHANNELS]
                [--warmup_steps WARMUP_STEPS] [--beta0 BETA0] [--beta1 BETA1] [--beta2 BETA2] [--filter_empty | --no-filter_empty]
                [--model_name MODEL_NAME] [--logs LOGS]

options:
  -h, --help            show this help message and exit
  --epochs EPOCHS       Max number of epochs.
  --projected_channels PROJECTED_CHANNELS
                        Number of channels in the immediate space projected through the stem (and conditional coder).
  --z_channels Z_CHANNELS
                        Number of channels in the latent space at each layer.
  --warmup_steps WARMUP_STEPS
                        Number of steps for KL divergence linear deterministic warmup.
  --beta0 BETA0         Beta value for KL divergence corresponding to layer 0 (shallowest layer).
  --beta1 BETA1         Beta value for KL divergence corresponding to layer 1.
  --beta2 BETA2         Beta value for KL divergence corresponding to layer 2 (topmost layer).
  --filter_empty, --no-filter_empty
                        If set, filter out empty masks.
  --model_name MODEL_NAME
                        Directory name of saved model checkpoints and metadata.
  --logs LOGS           Root save directory for logs.
```

#### Testing

```sh
python -m arch.nvae.test -h

usage: test.py [-h] --model_path MODEL_PATH [--logs LOGS]

options:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        Path to model checkpoint.
  --logs LOGS           Root save directory for logs.
```

See `analysis/nvae` for further analysis on trained models.

### U-Net

As an extension of the main work on NVAE, an application involves using its
learned latent spaces as a shape prior in the objective function of U-Net to
improve segmentation quality. This anatomically constrained U-Net (ACU-Net)
takes in a cardiac scan as input and outputs a segmentation mask. The code is
located in `arch/unet/` and the entry points are `train.py` and `test.py`.

The baseline is a U-Net model trained with a cross-entropy objective without the
shape prior.

In this repository, experiments are conducted primarily with the ACDC dataset. The U-Net environment also supports the M&Ms dataset for domain adaptation and few-short learning experiments.

#### Example

Example for ACDC dataset.

```sh
# View data samples
python -m utils.data_viewer --dataset acdc --show_scans
# Train an ACU-Net model with good configurations (~60 minutes)
python -m arch.unet.train --augment
# Test (~5 minutes)
# A typical checkpoint path is:
# logs/unet_acdc/version_0/checkpoints/epoch=45-step=4922.ckpt
python -m arch.unet.test --model_path path/to/unet/checkpoint.ckpt
```

Example for M&Ms dataset.

```sh
# View data samples
python -m utils.data_viewer \
    --dataset mnms \
    --centre 1 \
    --num_subjects 5 \
    --sort_by_validity
# Train an ACU-Net model with good configurations
python -m arch.unet.train \
    --epochs 50 \
    --dataset mnms \
    --centre 1 \
    --num_subjects 5 \
    --sort_by_validity \
    --augment
# Test (~5 minutes)
# A typical checkpoint path is:
# logs/unet_mnms/version_0/checkpoints/epoch=25-step=156.ckpt
python -m arch.unet.test \
    --model_path path/to/unet/checkpoint.ckpt \
    --dataset mnms \
    --centre 1
```

Use [TensorBoard](#tensorboard) to see the train graphs and test metrics.

#### Training

```sh
python -m arch.unet.train -h

usage: train.py [-h] [--epochs EPOCHS] [--loss_reg {cross_entropy,shape_prior}] [--alpha ALPHA] [--dataset {acdc,mnms}] [--centre {1,2,3,4,5}] [--num_subjects NUM_SUBJECTS]
                [--sort_by_validity | --no-sort_by_validity] [--filter_empty | --no-filter_empty] [--augment | --no-augment] [--seed SEED] [--model_name MODEL_NAME] [--logs LOGS]
                [--pretrained_model_path PRETRAINED_MODEL_PATH]

options:
  -h, --help            show this help message and exit
  --epochs EPOCHS       Max number of epochs.
  --loss_reg {cross_entropy,shape_prior}
                        Regulariser technique.
  --alpha ALPHA         If using shape prior loss, the weight of cross entropy loss.
  --dataset {acdc,mnms}
                        Which dataset to use.
  --centre {1,2,3,4,5}  If using M&Ms and set, only use scans from the specified centre.
  --num_subjects NUM_SUBJECTS
                        Few-shot learning for M&Ms: Number of subjects to use. If -1, use all subjects.
  --sort_by_validity, --no-sort_by_validity
                        Few-shot learning for M&Ms: If set, use subjects with highest anatomical validity.
  --filter_empty, --no-filter_empty
                        If set, filter out empty masks.
  --augment, --no-augment
                        If set, augment training data with random flips.
  --seed SEED           Seed for train reproducibility. This only affects training, not data split.
  --model_name MODEL_NAME
                        Directory name of saved model checkpoints and metadata.
  --logs LOGS           Root save directory for logs.
  --pretrained_model_path PRETRAINED_MODEL_PATH
                        If set, load a pretrained model from this path and continue training.
```

#### Testing

```sh
python -m arch.unet.test -h

usage: test.py [-h] --model_path MODEL_PATH [--dataset {acdc,mnms}] [--centre {1,2,3,4,5}] [--logs LOGS]

options:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        Path to model checkpoint.
  --dataset {acdc,mnms}
                        Which dataset to use.
  --centre {1,2,3,4,5}  If using M&Ms and set, only use scans from the specified centre.
  --logs LOGS           Root save directory for logs.
```

See `analysis/unet` for further analysis on trained models.

### SimCLR

A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)
is used to pretrain a ResNet-18 model that acts to replace Inception-v3 in the
FID metric. This is motivated by FID being a weak, unstable metric for
non-natural, segmentation maps. We have pretrained the model for a newly
proposed metric, Frechet ResNet Distance with SimCLR (FRDS).

#### Example

```sh
# View data samples
python -m utils.data_viewer --dataset acdc --augment_simclr
# Pretrain a ResNet-18 model with good configurations for FRDS (~90 minutes)
python -m arch.simclr.train \
    --epochs 200 \
    --model_name frds-resnet-18
# Run the benchmark tests (~5 minutes)
# A typical checkpoint path is:
# logs/simclr_acdc/version_0/checkpoints/epoch=199-step=1400.ckpt
python -m arch.simclr.test --model_path path/to/simclr/checkpoint.ckpt
```

Use [TensorBoard](#tensorboard) to see the train graphs.

#### Training

```sh
python -m arch.simclr.train -h

usage: train.py [-h] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--model_name MODEL_NAME] [--logs LOGS]

options:
  -h, --help            show this help message and exit
  --epochs EPOCHS       Max number of epochs.
  --batch_size BATCH_SIZE
                        Batch size as defined by number of pairs.
  --model_name MODEL_NAME
                        Directory name of saved model checkpoints and metadata.
  --logs LOGS           Root save directory for logs.
```

#### Testing

```sh
python -m arch.simclr.test -h             
usage: test.py [-h] [--model_path MODEL_PATH] [--use_inception | --no-use_inception] [--logs LOGS]
               [--show_preview | --no-show_preview]

options:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        Path to model checkpoint.
  --use_inception, --no-use_inception
                        If set, use Inception-v3 and compute FID.
  --logs LOGS           Root save directory for logs.
  --show_preview, --no-show_preview
                        If set, show effect of the various disturbances only
                        and do not run tests. The visualisations are saved in
                        the out directory.
```

See `analysis/simclr` for further analysis on trained models.

### TensorBoard

TensorBoard allows visualisation of graphs and metrics from the train/test
process.

```sh
# VAE logs
tensorboard --logdir logs/vae_acdc
# NVAE logs
tensorboard --logdir logs/nvae_acdc
# U-Net logs
tensorboard --logdir logs/unet_acdc
```

## Acknowledgements

This project is authored by Freddy Jiang and supervised by Prof. Elsa Angelini
and Prof. Loïc Le Folgoc.

Data is sourced from the Automated Cardiac Diagnosis Challenge (ACDC)[1] and the Multi-Centre, Multi-Vendor & Multi-Disease Cardiac Image Segmentation
Challenge (M&Ms)[2, 3].

- https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html
- https://www.ub.edu/mnms/

The implementation of Nouveau VAE in this repository is written from scratch in PyTorch Lightning and is based on the official implementation ([codebase][nvae-official])[4].

[nvae-official]: https://github.com/NVlabs/NVAE

[1]: Bernard O, Lalande A, Zotti C, Cervenansky F, Yang X, Heng PA, et al. Deep
learning techniques for automatic MRI cardiac multi-structures segmentation and
diagnosis: is the problem solved? IEEE transactions on medical imaging.
2018;37(11):2514-25.

[2]: Campello VM, Gkontra P, Izquierdo C, Martin-Isla C, Sojoudi A, Full PM, et
al. Multi-centre, multi-vendor and multi-disease cardiac segmentation: the M&Ms
challenge. IEEE Transactions on Medical Imaging. 2021;40(12):3543-54.

[3]: Martín-Isla C, Campello VM, Izquierdo C, Kushibar K, Sendra-Balcells C,
Gkontra P, et al. Deep learning segmentation of the right ventricle in cardiac
MRI: the M&Ms challenge. IEEE Journal of Biomedical and Health Informatics.
2023;27(7):3302-13.

[4]: Vahdat A, Kautz J. NVAE: A deep hierarchical variational autoencoder.
Advances in neural information processing systems. 2020;33:19667-79.
