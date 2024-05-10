from matplotlib import pyplot as plt
import os
import pandas as pd

from const import ACDC, LOGS_PATH

if __name__ == '__main__':
    plt.style.use("ggplot")

    paths = [
        os.path.join(LOGS_PATH, ACDC.DIR.VAE, "beta-vae.csv"),
        os.path.join(LOGS_PATH, ACDC.DIR.VAE, "info-vae.csv"),
        os.path.join(LOGS_PATH, ACDC.DIR.VAE, "info-adversarial-vae.csv"),
    ]

    labels = ["beta-VAE", "InfoVAE", "InfoAdversarialVAE"]

    for label, path in zip(labels, paths):
        df = pd.read_csv(path, index_col="model_name")
        # Filter all rows with fid >= 150
        df = df[df["fid"] < 150]
        plt.scatter(df["fid_manual"], df["fid"], alpha=0.5, label=label)
    
    plt.xlabel("FID Manual")
    plt.ylabel("FID")
    
    plt.legend()
    plt.show()
