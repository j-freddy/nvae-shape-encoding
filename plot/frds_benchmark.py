from matplotlib import pyplot as plt

# Collected data from FRDS benchmarking with simclr/test.py
benchmark_data = {
    "labels": ["v4", "v2-no-elastic", "Inception", "v3", "v3-small-elastic"],
    "ideal": [11.65, 12.66, 4.48, 18.08, 18.47],
    "disturbances": {
        "smoothing": {
            "labels": ["3x3", "5x5", "7x7", "9x9"],
            "data": [
                [11.48, 11.03, 10.50, 11.23],
                [14.08, 14.60, 15.06, 15.55],
                [7.42, 8.15, 8.47, 8.57],
                [19.37, 19.53, 19.34, 18.72],
                [21.91, 22.66, 22.93, 22.81],
            ],
        },
        "black box crop": {
            "labels": ["10-30%", "20-50%", "30-70%", "40-90%"],
            "data": [
                [17.88, 74.66, 187.73, 358.55],
                [14.77, 26.66, 76.24, 194.04],
                [6.02, 21.19, 61.98, 123.84],
                [9.64, 43.50, 155.28, 312.10],
                [12.30, 42.61, 142.81, 288.46],
            ],
        },
        "elastic deformation": {
            "labels": ["8", "6", "4", "2"],
            "data": [
                [25.14, 39.73, 198.07, 2489.88],
                [17.53, 37.73, 307.75, 3743.85],
                [10.13, 16.01, 43.66, 140.39],
                [15.13, 27.83, 131.76, 1354.71],
                [17.99, 35.64, 317.87, 3429.04],
            ],   
        },
        "pepper noise": {
            "labels": ["0.0005", "0.005", "0.05", "0.5"],
            "data": [
                [11.55, 11.39, 63.59, 1570.55],
                [13.10, 22.26, 275.16, 1523.90],
                [4.74, 16.82, 31.26, 32.93],
                [18.71, 24.82, 63.83, 880.50],
                [18.64, 21.26, 75.79, 2167.27],
            ],
        },
    },
}

if __name__ == '__main__':
    plt.style.use("ggplot")
    
    model_labels = benchmark_data["labels"]
    
    # For specific model
    for model_idx, model_name in enumerate(model_labels):
        fig, axs = plt.subplots(2, 2)
        fig_idx = 0
        
        # For each disturbance
        for disturbance_label, disturbance_data in benchmark_data["disturbances"].items():
            ax = axs[fig_idx // 2, fig_idx % 2]
            
            values = disturbance_data["data"][model_idx]
            
            # Plot ideal value with other values
            agg_values = [benchmark_data["ideal"][model_idx]] + values
            labels = ["none"] + disturbance_data["labels"]
            ax.plot(labels, agg_values)
            
            metric = "FID" if model_name == "Inception" else "FRDS"
            
            ax.set(xlabel=disturbance_label, ylabel=metric)
            
            fig_idx += 1
        
        print(f"Showing {model_name}")
        
        fig.suptitle(f"Model: {model_name}")
        fig.tight_layout()
        plt.show()
