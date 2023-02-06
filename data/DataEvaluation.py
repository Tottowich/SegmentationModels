"""
This is a file conaining functions to evaluate the various datasets
Extract valuable information from the dataset and save it to a file in csv format
The results should be save to a folder called "results/'dataset_name'/"
The results should be saved in a file called "results.csv" along with plots of distrubutions and other useful information
"""
import numpy as np
import pandas as pd
import os
import yaml
import matplotlib.pyplot as plt
import plotly.graph_objects as go # Plotly for interactive plots

import pandas as pd
import numpy as np
import os
import yaml
import torch as T
from tqdm import tqdm
# Import dataloader
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Dict, Union, Optional, Type, Any

class DataMetric:
    # A parent class for all the metrics, this class should be inherited by all the metrics
    # All the metrics should have a name.
    # All metrics should have a function to calculate the metric
    # All metrics should have a function to plot the metric
    # All metric should store contain a set of values for the metric
    # All metrics should contain a function to summarize the set of values
    def __init__(self, name: str):
        self.name = name
        self.values = []
    def calculate(self, *args, **kwargs):
        raise NotImplementedError
    def plot(self, *args, **kwargs):
        raise NotImplementedError
    def summarize(self, *args, **kwargs):
        raise NotImplementedError  

class LabelDistribution(DataMetric):
    # A class to store the distribution of the data
    def __init__(self, name: str, num_classes: int = 10,class_names:List[str]=None, *args, **kwargs):
        super().__init__(name)
        self.num_classes = num_classes
        self.class_names = class_names if class_names is not None else [f"class_{i}" for i in range(num_classes)]
        self.mean = None
    def calculate(self, data: np.ndarray):
        # Calculate the histogram of the data
        # The input data has shape (N,num_classes,W,H)
        # The output data has shape (num_classes,)

        # Total number of pixels
        total_pixels = np.prod(data.shape)
        # Calculate the histogram
        hist, bin_edges = np.histogram(data, bins=self.num_classes)
        # Normalize the histogram
        hist = hist / total_pixels
        # Append the histogram to the values
        self.values.append(hist)
        return hist
    # Plot should plot and return the figure
    def plot(self, *args, **kwargs)->plt.Figure:
        # Plot the distribution of the data as a histogram and as a pie diagram
        if not hasattr(self, "mean"):
            self.summarize()
        # Plot the histogram
        fig, ax = plt.subplots(1,2)
        # Class names should be at an angle on the x-axis
        ax[0].bar(np.arange(self.num_classes), self.mean)
        ax[0].set_title("Histogram of the distribution")
        ax[0].set_xlabel("Class")
        ax[0].set_ylabel("Frequency")
        # Plot the pie diagram
        # Hide the labels for the pie diagram, TODO: Add labels on hover
        ax[1].pie(self.mean, labels=["" for i in range(self.num_classes)])
        # Get tooltips for the pie diagram
        # tooltips = [f"{self.class_names[i]}: {self.mean[i]:.2f}" for i in range(self.num_classes)]
        # Add tooltips to the pie diagram
        ax[1].set_title("Pie diagram of the distribution")
        plt.show()
        return fig
    def summarize(self, *args, **kwargs)->tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Summarize the distribution by calculating the mean and standard deviation
        # Return the mean and standard deviation
        self.mean = np.mean(self.values, axis=0)
        self.std = np.std(self.mean, axis=0)
        self.ranking = np.argsort(self.mean)
        return self.mean, self.std, self.ranking

class PositionalDistribution(DataMetric):
    """
    A class to store and calculate the positional distribution
    of the dataset.
    """
    def __init__(self, name: str, num_classes: int = 10,img_size: Tuple[int,int] = (224,224), class_names:List[str]=None,*args, **kwargs):
        super().__init__(name)
        self.num_classes = num_classes
        self.img_size = img_size
        # The positional distribution is a 2D histogram
        # The first dimension is the class
        # The second and third dimension is the position
        self.totals = np.zeros((num_classes, *img_size))
        self.eye = np.eye(num_classes,dtype=np.uint8)
        self.class_names = class_names if class_names is not None else [f"class_{i}" for i in range(num_classes)]
    def calculate(self, data: np.ndarray)->np.ndarray:
        # The input data has shape (N,W,H)
        # The output data has shape (num_classes,W,H)
        # Convert the data to a one-hot encoding
        one_hot = self.eye[data]
        # Sum the one-hot encoding over the first dimension
        # This gives the total number of pixels for each class
        totals = np.sum(one_hot, axis=0, dtype=np.uint32)
        self.totals += totals.transpose(2,0,1)
        # Append the totals to the values
        self.values.append(totals)
        return totals
    def summarize(self):
        # Get the most common class for each pixel
        self.ranking = np.argmax(self.totals, axis=0)
        # Calculate the probability of each class for each pixel
        # Get the total number of pixels for each class
        total_pixels = np.sum(self.totals, axis=(-2,-1))
        # print(f"The total number of pixels for each class is {total_pixels}, shape: {total_pixels.shape}")
        per_class_total = total_pixels.reshape(151,1,1)
        per_class_total = np.repeat(per_class_total,self.img_size[0],axis=1)
        per_class_total = np.repeat(per_class_total,self.img_size[0],axis=2)
        per_class_total = np.where(per_class_total==0,1,per_class_total)
        self.probability = self.totals/per_class_total

        # self.probability = self.totals / total_pixels#np.sum(self.totals, axis=0)
        return self.ranking, self.probability
    def plot(self, *args, **kwargs)->plt.Figure:
        # Plot the positional distribution
        if not hasattr(self, "ranking"):
            self.summarize()
        # Create an interactive plot using plotly
        # The plot should have a dropdown menu to select the class
        # Import plotly and create the figure with the dropdown menu
        # Should also have a subplot for the ranking of the classes


        fig = go.Figure()
        # Add the dropdown menu
        fig.update_layout(
            updatemenus=[
                dict(
                    active=0,
                    buttons=list([
                        dict(label=f"#{i}: "+self.class_names[i],
                            method="update",
                            args=[{"visible": [ i==j for j in range(self.num_classes+1)]}])
                        for i in range(self.num_classes)
                    ])+[dict(label="Ranking",
                    method="update", 
                    args=[{"visible": [False for i in range(self.num_classes)]+[True]}])],
                )
            ]
        )
        # Add the positional distribution for each class
        for i in range(self.num_classes):
            fig.add_trace(go.Heatmap
                (
                    z=self.probability[i],
                    name=self.class_names[i],
                    visible=i == 0,
                )
            )
        # Add ranking figure to the plot
        fig.add_trace(
            go.Heatmap(
                z=self.ranking,
                name="Ranking",
                visible=False,
                colorscale="Viridis",
                # Hover with ranking and class name
                hovertemplate="Ranking: %{z}<br>Class: %{customdata[z]}<extra></extra>",
                # hovertemplate="Ranking: %{z}<br><extra></extra>",#Class: %{customdata}<extra></extra>",
                customdata=self.class_names
            )
        )
        fig.update_layout(
            title="Positional distribution of the classes",
            xaxis_title="X",
            yaxis_title="Y",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            )
        )

    
        # Show the plot
        fig.show()
        return fig    
# A Class to evaluate the dataset
class DatasetEvaluator:
    """
    A class to evaluate the dataset, and store the results.
    The results are stored in a pandas dataframe and can easily be visualized.
    
    The evaluation is done using the DataMetric classes.
    The DataMetric classes are stored in a dictionary, where the key is the name of the class.
    
    """
    __metric_classes = [cls for cls in DataMetric.__subclasses__()]
    metrics = {metric.__name__: metric for metric in __metric_classes}
    def __init__(self, dataset_name: str, dataset:Dataset,results_dir: str,dataloader:DataLoader=None,class_names:list[str]=None,metrics:list[str]=None):
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.dataloader = dataloader
        self._validate_dataset()
        self.results_dir = results_dir
        self.class_names = class_names
        self._validate_classes()
        self.num_classes = len(self.class_names)
        self._create_metrics(metrics)
        # Use a pandas dataframe to store the results
        self.results = pd.DataFrame()
        # Create the results directory if it does not exist
        self._create_dir()
    def _validate_dataset(self):
        # Check if the dataset is a valid dataset
        assert isinstance(self.dataset, Dataset), "The dataset must be a valid dataset"
        if not self.dataloader:
            batch_size = 100
            print(f"Creating a dataloader since the dataset is not a dataloader. Batch size: {batch_size}")
            self.dataloader = DataLoader(self.dataset, batch_size=batch_size)
    def _create_metrics(self,metrics)->List[DataMetric]:
        # Create the metrics
        self.data_metrics =[]
        if metrics is None:
            metrics = self.metrics.keys()
        for metric in metrics:
            print(f"Creating metric {metric}")
            if metric in self.metrics:
                self.data_metrics.append(self.metrics[metric](metric, self.num_classes,class_names=self.class_names))
            else:
                raise ValueError(f"Metric {metric} does not exist")
        
        return self.metrics
    def _validate_classes(self):
        # Validate the classes
        if self.class_names is None:
            assert hasattr(self.dataset, "class_names"), "The dataset does not have a classes attribute"
            self.class_names = self.dataset.class_names
    def _create_dir(self):
        # Create the results directory if it does not exist
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        # Create a directory for the dataset
        if not os.path.exists(os.path.join(self.results_dir, self.dataset_name)):
            os.mkdir(os.path.join(self.results_dir, self.dataset_name))        

    def evaluate(self):
        # Evaluate the data
        pbar = tqdm(self.dataloader, desc="Evaluating dataset")
        for _, label in pbar:
            label = np.argmax(label.numpy(), axis=1)
            # Argmax the data to a (N,H,W) shape
            for metric in self.data_metrics:
                metric.calculate(label)
        # Summarize the metrics
        self.results = {}
        for metric in self.data_metrics:
            values = metric.summarize()
            self.results[metric.name] = values
        return self.results
    def get_results(self):
        # Get the results
        return self.results
    def plot(self, *args, **kwargs)->list[plt.Figure]:
        # Plot the results
        # Create a figure
        # fig, axes = plt.subplots(nrows=len(self.data_metrics), ncols=1, figsize=(10, 10))
        # Plot the results
        figs = []
        for i, metric in enumerate(self.data_metrics):
            figs.append(metric.plot())
        # Return the figure
        return figs
    def interactive_plot(self):
        # The plot should have three columns.
        # The first column is the image of the dataset.
        # The second column is the mask of the dataset.
        # The third column is a table with different results based on the pixel values of the mask.
        # The table should have the following columns:
        # - Pixel value = label name
        # - Proportion of pixels in the entire dataset with that label.
        # - Occurrences of the label in the dataset, i.e. how many images contain that label.
        # - Proportion of pixels in the dataset with that label, but only in the images that contain that label.

        # The figure should have a dropdown menu to select the image to show. The dropdown menu should have the following options:
        # - Image index.
        # - Random image, button.
        # - Next image, button.
        # - Previous image, button.
        # - First image, button.
        # - Last image, button.

        # The figure is created using plotly

        # Create the subplot figure
        fig = go.Figure().set_subplots(rows=1, cols=2, specs=[[{"type": "image"}, {"type": "heatmap"}]])

        for i, (image, label) in enumerate(self.dataset):
            # Create the aligned images
            label = label.argmax(dim=0)
            label_image = go.Heatmap(
                z=np.flipud(label.numpy()).astype(np.int32),
                name=i,
                visible=False,
                # Set origin to the top left corner,
                # The color scale
                colorscale="Viridis",
                # Remove the colorbar
                showscale=False,

                hovertemplate="Ranking: %{z}<br>Class: %{customdata[z]}<extra></extra>",
                customdata=self.class_names
            )
            image_image = go.Image(
                z=image.numpy().transpose(1,2,0)*256,
                name=i,
                visible=False,
            )
            # Add the frame to the figure
            fig.add_trace(image_image, row=1, col=1)
            # Add the label image correctly orientated
            fig.add_trace(label_image, row=1, col=2)
        # Make the first frame visible
        fig.data[0].visible = True
        fig.data[1].visible = True

        # Add the slider
        steps = []
        for i in range(min(100, len(self.dataset))):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(fig.data)}],
            )
            step["args"][0]['visible'][2*i] = True  # Toggle i'th trace to "visible"
            step["args"][0]['visible'][2*i+1] = True  # Toggle i'th trace to "visible"
            steps.append(step)
        sliders = [dict(active=0, currentvalue={"prefix": "Frame: "}, pad={"t": 0}, steps=steps)]
        fig.update_layout(
            sliders=sliders
        )

        # Show the figure
        fig.show()
    def save(self):
        raise NotImplementedError
        
        