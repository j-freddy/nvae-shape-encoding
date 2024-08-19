import pandas as pd

if __name__ == "__main__":
    path = "logs/unet_mnms/no-finetune.csv"
    df = pd.read_csv(path, index_col="model_name")

    labels = []
    labels.append(f"shape-prior")

    for label in labels:
        df_centre = df[df.index.str.startswith(label)]
        
        # Take the average of each column
        avg = df_centre.mean()
        
        # Compute standard deviation
        std = df_centre.std()
        
        # Report the results to 3 significant figures
        print(label)
        print(avg.round(6))
        print(std.round(6))
        print("\n")
