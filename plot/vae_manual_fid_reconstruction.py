from matplotlib import pyplot as plt
import os
import pandas as pd

from const import ACDC, LOGS_PATH

if __name__ == '__main__':
    plt.style.use("ggplot")

    paths = [
        os.path.join(LOGS_PATH, ACDC.DIR.VAE, "info-vae.csv"),
        os.path.join(LOGS_PATH, ACDC.DIR.VAE, "info-vae-register.csv"),
    ]

    labels = ["InfoVAE", "InfoVAE-Register-Alignment"]

    for label, path in zip(labels, paths):
        df = pd.read_csv(path, index_col="model_name")

        # Filter all rows with fid >= 25
        df = df[df["fid_manual"] < 25]

        plt.scatter(df["fid_manual"], df["test_recon_loss"], alpha=0.5, label=label)
    
    plt.xlabel("FID Manual")
    plt.ylabel("Reconstruction Loss")
    
    plt.legend()
    plt.show()
