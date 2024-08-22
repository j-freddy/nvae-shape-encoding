import pandas as pd

if __name__ == "__main__":
    path = "logs/unet_acdc/best.csv"
    df = pd.read_csv(path, index_col="model_name")

    labels = ["baseline", "shape-prior"]

    for label in labels:
        df_centre = df[df.index.str.startswith(label)]
        
        # Take the average of each column
        avg = df_centre.mean()
        
        # Compute standard error
        std = df_centre.std() / (df_centre.count() ** 0.5)
        
        # Report the results to 3 significant figures
        print(label)
        print(avg.round(3))
        print(std.round(3))
        print("\n")
