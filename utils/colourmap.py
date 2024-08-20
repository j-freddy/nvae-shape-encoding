from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np

def get_giridis():
    """
    Viridis with grey background
    """
    viridis = plt.cm.viridis
    viridis_colours = viridis(np.arange(viridis.N))
    viridis_colours[0] = colors.to_rgba("#212529")
    return colors.ListedColormap(viridis_colours)

def get_gred():
    """
    Red with grey background
    """
    colours = ["#212529", "#ff6b6b"]
    return colors.ListedColormap(colours)

GIRIDIS = get_giridis()
GREDS = get_gred()
