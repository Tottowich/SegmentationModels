# Tools for visualizing the results of the segmentation as well as attention maps
import numpy as np
import torch.nn.functional as F
from typing import Callable
from matplotlib import cm
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
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

class SegmentationVisualizer:
    """
    Visualize the segmentation results
    """
    def __init__(self, n_classes:int, class_names:list[str]=None,display:bool=True):
        self.n_classes = n_classes
        self.class_names = class_names
        self.annotations = []
        self.font_size = 15
        self.display = display
        self.text_box_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9,facecolor="wheat")
        # Each class should be assigned a color
        self.colors = cm.rainbow(np.linspace(0, 1, n_classes))
        # The background class should be black
        self.colors[-1] = [0, 0, 0,0]
    def grid(self, images:np.ndarray, masks:np.ndarray, predictions:np.ndarray=None)-> plt.Figure:
        """
        Visualize the segmentation results in a grid
        """
        # Create a figure with a grid of subplots
        figs = 3 if predictions is not None else 2
        batch_size = images.shape[0] if len(images.shape) == 4 else 1
        if predictions is None:
            predictions = [None] * batch_size
        # Create a figure with individual axes for each image

        if batch_size > 5:
            rows = batch_size//5
            rows = rows + 1 if batch_size % 5 != 0 else rows
            fig, axes = plt.subplots(rows, 5, figsize=(15, 15))
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(1, batch_size, figsize=(10, 10))
            if batch_size == 1:
                axes = [axes]
        # Remove the axes
        for ax in axes:
            ax.axis("off")
        # Add the images
        for i, (ax, image,mask,prediction) in enumerate(zip(axes, images, masks, predictions)):
            ax.imshow(images[i].transpose(1, 2, 0))
            ax.imshow(masks[i].squeeze(), cmap="viridis", alpha=0.5)
            if prediction is not None:
                ax.imshow(prediction.squeeze(), cmap="viridis", alpha=0.5)
        text_dict = {}
        # Add axes to keys for easy access
        for i, ax in enumerate(axes):
            text_dict[ax] = ax.text(0.5,1.0, "", transform=ax.transAxes, fontsize=self.font_size,
                                    ha="center", bbox=self.text_box_props)
        def on_move(event):
            x, y = event.xdata, event.ydata
            if event.inaxes:
                ax = event.inaxes
                for i, ax in enumerate(axes):
                    if ax == event.inaxes:
                        # Get the pixel coordinates
                        # Transform from display coordinates to data coordinates
                        int_x, int_y = int(x), int(y)
                        if int_x < 0 or int_y < 0 or int_x >= images[i].shape[1] or int_y >= images[i].shape[2]:
                            return
                        label, color = self.mask_pred_label(int_x, int_y, masks[i], predictions[i])
                        # Add text box above ax
                        text_dict[ax].set_text(label)
                        text_dict[ax].set_color(color)
                        # txt = ax.text(0.5, 1.1, label, color=color,
                        #       fontsize=self.font_size, bbox=self.text_box_props,ha="center",transform=ax.transAxes)
                    
                        fig.canvas.draw_idle()
                    
        # Add the masks
        # Add the predictions
        # Return the plot
        # if self.display:
        #     self.show()
        fig.tight_layout(pad=0.0,w_pad=0.0,h_pad=1)
        fig.canvas.mpl_connect('motion_notify_event', on_move)
        plt.show()
        return fig

        self.fig = plt.figure(figsize=(10, 10))
        grid = ImageGrid(self.fig, 111,  # similar to subplot(111)
                         nrows_ncols=(batch_size, figs),  # creates 2x2 grid of axes
                         )
        if predictions is None:
            predictions = [None] * batch_size
        # Remove axes
        for ax in grid:
            ax.axis("off")
        # Add the images, masks, and predictions to the grid
        for i,(image,mask,prediction) in enumerate(zip(images, masks, predictions)):
            # Add the image
            grid[i*figs].imshow(image.transpose(1, 2, 0))
            # Add the mask
            grid[i*figs+1].imshow(mask.squeeze(), cmap="viridis")
            # Add the prediction
            if prediction is not None:
                grid[i*figs+2].imshow(prediction.squeeze(), cmap="viridis")
            # grid[i*3+2].imshow(prediction.squeeze(), cmap="viridis")
        # Return the plot
        if self.display:
            self.show()
        return self.fig
    def visualize_predictions(self,image,mask,prediction)-> plt.Figure:
        # Create a subplot containing image, mask and prediction
        self.fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 15),sharex=True)
        # Remove the axes
        for ax in (ax1, ax2, ax3):
            ax.axis("off")
        # Add the image
        ax1.imshow(image.transpose(1, 2, 0))
        # Add the mask
        ax2.imshow(mask.squeeze(), cmap="viridis")
        # Add the prediction
        ax3.imshow(prediction.squeeze(), cmap="viridis")
        # Return the plot
        # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        def on_move(event):
            x, y = event.x, event.y
            if event.inaxes:
                ax = event.inaxes  # the axes instance
                # pred_id = prediction[int(event.ydata), int(event.xdata)]
                # pred_name = self.class_names[pred_id].split(",")[0]
                # mask_id = mask[int(event.ydata), int(event.xdata)]
                # mask_name = self.class_names[mask_id].split(",")[0]
                # color = "red" if pred_id != mask_id else "green"
                label, color = self.mask_pred_label(int(event.ydata), int(event.xdata), mask, prediction)
                txt_main = self.fig.text(0.5, 1.1, label, color=color,
                              fontsize=15, bbox=self.text_box_props,ha="center",transform=ax2.transAxes)
                
                self.annotations.append((None, txt_main))
                if len(self.annotations)>1:
                    txt, txt_main = self.annotations.pop(0)
                    if txt is not None:
                        txt.remove()
                    if txt_main is not None:
                        txt_main.remove()

                self.fig.canvas.draw()
        if self.display:
            self.show(on_move=on_move)
        return self.fig
    def mask_pred_label(self, int_x,_int_y, mask, prediction)->tuple[str,str]:
        if prediction is not None:
            pred_id = prediction[int(_int_y), int(int_x)]
            pred_name = self.class_names[pred_id].split(",")[0]
        mask_id = mask[int(_int_y), int(int_x)]
        mask_name = self.class_names[mask_id].split(",")[0]
        if prediction is not None:
            color = "red" if pred_id != mask_id else "green"
            return f"pred: {pred_name} ({pred_id})\nmask: {mask_name} ({mask_id})", color
        else:
            return f"mask: {mask_name} ({mask_id})", "green"
    def show(self,name="Figure",on_move:Callable=None):
        # plt.close()
        if on_move is not None:
            self.fig.canvas.mpl_connect('motion_notify_event', on_move)
        self.fig.show("Figure")
if __name__=="__main__":

    visualizer = SegmentationVisualizer(n_classes=151,class_names=dataset.class_names)
    visualizer.grid(trainx,trainy,prediction)
    visualizer.visualize_predictions(image.numpy(),mask.numpy(),prediction[0].numpy())

