import pandas as pd

if __name__ == "__main__":
    path = "logs/unet_mnms/shape-prior.csv"
    df = pd.read_csv(path, index_col="model_name")

    # Filter out rows that start with "centre-1"

    for centre_id in range(1, 6):
        df_centre = df[df.index.str.startswith(f"centre-{centre_id}")]
        
        # Take the average of each column
        avg = df_centre.mean()
        
        # Report the results to 3 significant figures
        print(f"Centre {centre_id}")
        print(avg.round(3))
        print("\n")
