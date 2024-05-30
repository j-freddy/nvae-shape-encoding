from matplotlib import pyplot as plt
import os
import pandas as pd

from const import ACDC, LOGS_PATH

def filter_entries(dfs: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """
    Hyperparameter tuning of different models may have different grid search
    values. This function filters out all entries such that the resulting
    dataframes have the same configurations. This allows for a fair comparison
    during plotting.
    
    Example: - Before
        - InfoVAE:         gamma-1, gamma-10, gamma-100, gamma-1000
        - InfoVAE Augment: gamma-1, gamma-1000
    - After
        - InfoVAE:         gamma-1, gamma-1000
        - InfoVAE Augment: gamma-1, gamma-1000
    
    noqa: Perform filtering by matching index column (i.e. model_name) between
    all dataframes.
    """
    configs = pd.Index([])
    
    for df in dfs:
        if len(configs) == 0:
            configs = df.index
            continue

        configs = configs.intersection(df.index)
    
    filtered_dfs = []
    
    for df in dfs:
        filtered_dfs.append(df.loc[configs])
        
    return filtered_dfs

if __name__ == '__main__':
    plt.style.use("ggplot")

    # Customisable: Choose which metrics to plot from logs/vae_acdc
    paths = [
        os.path.join(LOGS_PATH, ACDC.DIR.VAE, "beta-vae.csv"),
        os.path.join(LOGS_PATH, ACDC.DIR.VAE, "info-vae.csv"),
    ]

    # Customisable: Ensure labels align with paths
    labels = ["beta-VAE", "InfoVAE"]
    
    # Read data
    dfs = []
    
    for path in paths:
        df = pd.read_csv(path, index_col="model_name")
        dfs.append(df)
    
    # dfs = filter_entries(dfs)

    for label, df in zip(labels, dfs):
        # Filter all rows with fid_resnet >= 100
        df = df[df["fid_resnet"] < 100]
        plt.scatter(df["fid_resnet"], df["test_recon_loss"], alpha=0.5, label=label)
    
    plt.xlabel("FID ResNet")
    plt.ylabel("Reconstruction Loss")
    
    plt.tight_layout()
    plt.legend()
    plt.show()
