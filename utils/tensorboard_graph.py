from matplotlib import pyplot as plt
import os
import pandas as pd

from utils.const import OUT_PATH

plt.style.use("ggplot")

def plot(path: str):
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
    plt.legend(["Top-5 Accuracy"])
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    name = "run-resnet-18-v2-no-elastic-tag-acc_top5.csv"
    path = os.path.join(OUT_PATH, "csv", name)
    plot(path)
