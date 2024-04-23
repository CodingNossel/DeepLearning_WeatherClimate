import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch
import os

def heat_plotting(tensor, title = "Temperatur", value = 0, level = 0):
    """Plotting a heatmap of a level with it's values in specific dimensions

    Args:
        tensor (torch.tensor): Tensor in the shape of (36,72,3,10).
        title (str, optional): The title of the Heatmap. Defaults to "Temperatur".
        value (int, optional): Of the Tensor shape the 3 - 3 is seperated into: 0 - Temp, 1 - Wind, 2 - Second Wind. Defaults to 0.
        level (int, optional): Of the Tensor shape the 10 - Level Highs. Defaults to 0.
    """
    sns.set_theme(style="darkgrid")
    df = pd.DataFrame(tensor[:,:,value,level].numpy())
    sns.heatmap(df)
    output_dir = "visualizations_output"
    os.makedirs(output_dir, exist_ok=True)
    plt.title(title)
    plt.savefig(os.path.join(output_dir, title + '_heatmap.png'))
    plt.close()
