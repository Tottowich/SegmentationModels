# Tools for visualizing the results of the segmentation as well as attention maps
import matplotlib.pyplot as plt
import numpy as np
import torch as T
import torch.nn.functional as F
from matplotlib import cm
from mpl_toolkits.axes_grid1 import ImageGrid
from .helper import to0_1



def visualize_attention_map(attention_map, image, save_path=None):
    """
    Visualize the attention map on top of the image
    """
    n_maps = attention_map.shape[0]
    # Make square grid of attention maps
    n_rows = int(np.ceil(np.sqrt(n_maps)))
    n_cols = int(np.ceil(n_maps / n_rows))
    fig = plt.figure(figsize=(n_cols, n_rows))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(n_rows, n_cols),
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    # Plot heatmaps with attention:
    img = to0_1(image).transpose(1, 2, 0)
    for ax,attn in zip(grid,attention_map):
        attn = to0_1(attn)

        ax.imshow(img)
        ax.imshow(to0_1(attn), alpha=0.5, cmap="magma")
        ax.axis("off")
    plt.show()

