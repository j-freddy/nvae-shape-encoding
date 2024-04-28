from arch.vae.factorvae import FactorVAE
from arch.vae.tcvae import TCVAE
from arch.vae.vae import VAE

ID_TO_MODEL = {
    "beta_vae": VAE,
    "beta_tcvae": TCVAE,
    "factor_vae": FactorVAE,
}
