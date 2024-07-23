import csv
import os
import pandas as pd
import requests

from utils.const import ACDC, LOGS_PATH

def get_tensorboard_data(log_dir: str, metrics: list[str]) -> pd.DataFrame:
    """
    Scrape single-value metrics from TensorBoard logs. TensorBoard must be
    active on localhost:6006.
    """
    experiments = os.listdir(log_dir)

    df = pd.DataFrame(columns=["model_name"] + metrics)
    df.set_index("model_name", inplace=True)

    for i, model_name in enumerate(experiments):
        value_buffer = []

        corrupted = False

        for metric in metrics:
            url = f"http://localhost:6006/data/plugin/scalars/scalars?tag={metric}&run={model_name}&format=csv"

            response = requests.get(url, allow_redirects=True)
            data_csv = csv.reader(response.text.splitlines(), delimiter=",")

            header = next(data_csv)
            
            if header[-1] != "Value":
                print(f"Corrupted data. Printing received data for {model_name}")
                print(header)
                print(list(data_csv))
                corrupted = True
                break
            
            value = float(next(data_csv)[-1])
            value_buffer.append(value)

        if not corrupted:
            df.loc[model_name] = value_buffer

        print(f"Current progress: {i + 1}/{len(experiments)}")
    
    return df

if __name__ == '__main__':
    # Customisable: Configure the folder and metrics to scrape
    log_subdir = "latent-skip"
    metrics = ["gen/frds", "gen/anatomically_valid", "loss/dsc",]

    df = get_tensorboard_data(
        log_dir=os.path.join(LOGS_PATH, ACDC.DIR.NVAE, log_subdir),
        metrics=metrics,
    )

    df.sort_values(by="loss/dsc", inplace=True, ascending=False)

    print(df.head())

    # Save dataframe to a csv file
    df.to_csv(os.path.join(LOGS_PATH, ACDC.DIR.NVAE, f"{log_subdir}.csv"))
