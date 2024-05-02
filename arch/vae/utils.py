from arch.vae.info_adversarial_vae import InfoAdversarialVAE
from arch.vae.info_vae import InfoVAE
from arch.vae.vae import VAE

ID_TO_MODEL = {
    "beta_vae": VAE,
    # TODO These IDs should be updated, but I have already trained the models
    # with the IDs as hparams. Do update them at some point.
    "beta_tcvae": InfoVAE,
    "factor_vae": InfoAdversarialVAE,
}
