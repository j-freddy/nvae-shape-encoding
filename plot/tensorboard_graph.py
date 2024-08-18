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
    # data = data[data["Step"] >= 2140]
    
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
    # data = [df[df["Step"] >= 2140] for df in data]
    
    for i, df in enumerate(data):
        steps = df["Step"].values
        values = df["Value"].values
        
        if i == 1:
            values = 126717 * values

        plt.plot(steps, values, label=labels[i])
    
    plt.xlabel("Step")
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_PATH, "figure.png"), dpi=400)
    plt.show()

if __name__ == '__main__':
    names = [
        "baseline.csv",
        "shape-prior.csv",
    ]
    
    labels = [
        "Cross-Entropy",
        r"126717 $\times$ Shape Prior",
    ]
    
    paths = [os.path.join(OUT_PATH, "csv", name) for name in names]
    plot_multiple(paths, y_label="Loss", labels=labels)
