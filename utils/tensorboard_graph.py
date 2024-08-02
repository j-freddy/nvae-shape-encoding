from matplotlib import pyplot as plt
import os
import pandas as pd

from utils.const import OUT_PATH

plt.style.use("ggplot")

def plot(path: str, y_label: str, label: str):
    """
    The TensorBoard graph has limited expressivity. Download the data in .csv
    format and plot it with matplotlib.
    """
    data = pd.read_csv(path)
    
    # TensorBoard keys
    steps = data["Step"].values
    values = data["Value"].values

    plt.plot(steps, values)
    plt.xlabel("Step")
    plt.ylabel(y_label)
    plt.legend([label])
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_PATH, "figure.png"), dpi=400)
    plt.show()

def plot_multiple(paths: list[str], y_label: str, labels: list[str]):
    """
    The TensorBoard graph has limited expressivity. Download the data in .csv
    format and plot it with matplotlib.
    """
    data = [pd.read_csv(path) for path in paths]
    data = [df[df["Step"] >= 2140] for df in data]
    
    for i, df in enumerate(data):
        steps = df["Step"].values
        values = df["Value"].values

        plt.plot(steps, values, label=labels[i])
    
    plt.xlabel("Step")
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_PATH, "figure.png"), dpi=400)
    plt.show()

if __name__ == '__main__':
    names = [
        "run-default-tag-loss_kl_div_group_6.csv",
        "run-default-tag-loss_kl_div_group_5.csv",
        "run-default-tag-loss_kl_div_group_4.csv",
        "run-default-tag-loss_kl_div_group_3.csv",
        "run-default-tag-loss_kl_div_group_2.csv",
        "run-default-tag-loss_kl_div_group_1.csv",
        "run-default-tag-loss_kl_div_group_0.csv",
    ]
    
    labels = [
        "Layer 1 Group 1",
        "Layer 2 Group 1",
        "Layer 2 Group 2",
        "Layer 3 Group 1",
        "Layer 3 Group 2",
        "Layer 3 Group 3",
        "Layer 3 Group 4",
    ]
    
    paths = [os.path.join(OUT_PATH, "csv", name) for name in names]
    plot_multiple(paths, y_label="KL Divergence", labels=labels)
