from arch.vae.info_adversarial_vae import InfoAdversarialVAE
from arch.vae.info_vae import InfoVAE
from arch.vae.vae import VAE

ID_TO_MODEL = {
    "beta_vae": VAE,
    "info_vae": InfoVAE,
    "info_adversarial_vae": InfoAdversarialVAE,
}
