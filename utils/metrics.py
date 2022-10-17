"""
Function and classes to gather metrics when training a model.
"""
import os
import time
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from typing import List, Tuple, Dict, Union, Optional, Any

    
class Metric(object):
    class_names = ["Background", *(f"Class {i}" for i in range(1, 150))]
    def __init__(self,name:str):
        self.name = name
        self.history = []

    def reset(self):
        """
        Reset the metric to its initial state.
        """
        self.history = []
        self._reset()
    def update(self, *args, **kwargs):
        """
        Update the metric with new data.
        """
        self._update(*args, **kwargs)
        self.history.append(self.value)
    def plot(self, value,writer:SummaryWriter=None,global_step:int=None,ax=np.ndarray,*args, **kwargs):
        """
        Plot the metric.
        """
        if value is None:
            raise ValueError("Metric has not been updated yet.")
        print(f"{self.name}: {value}")
        if writer is not None:
            writer.add_scalar(self.name,self.history[-1],global_step)
        if ax is not None:
            ax.set_data(range(len(self.history)),self.history)
    @property
    def value(self):
        pass
    @property
    def mean(self):
        pass

    # @property
    # def class_names(self):
    #     return self.__class__.class_names


class AverageMeter(Metric):
    def __init__(self, name="AverageMeter"):
        super(AverageMeter, self).__init__(name)
        self.reset()
    def _reset(self):
        self.sum = 0
        self.count = 0
    def _update(self, val:Union[int,float], n=1):
        self.sum += val * n
        self.count += n
    @property
    def value(self):
        return self.sum / self.count if self.count else 0

class AccuracyMeter(Metric):
    def __init__(self, name="AccuracyMeter",n_classes:int=1):
        super(AccuracyMeter, self).__init__(name)
        self.n_classes = n_classes
        self.correct_area = None
        self.reset()
    def _reset(self):
        self.correct = 0
        self.count = 0
        self._confusion_matrix = np.zeros((self.n_classes, self.n_classes), dtype=np.int32)
    def _update(self, pred:Union[T.Tensor, np.ndarray], target:Union[T.Tensor, np.ndarray]):
        """
        Predictions and targets are expected to be in the same shape.
        The predicitons should contain the class index.
        Updating the metric will update the confusion matrix.
        """
        if isinstance(pred, T.Tensor):
            pred = pred.detach().cpu().numpy().astype(np.int32)
        if isinstance(target, T.Tensor):
            target = target.detach().cpu().numpy().astype(np.int32)
        assert pred.shape == target.shape, "Predictions and targets should have the same shape." # Shape should be (batch_size, H, W)
        self.correct_area = pred == target
        self.correct += np.sum(self.correct_area)
        self.count += pred.size
        # Confusion matrix should be (n_classes, n_classes) and contain the fraction of pixels of class i that were predicted as class j.
        self._confusion_matrix += np.bincount(self.n_classes * target.reshape(-1) + pred.reshape(-1), minlength=self.n_classes**2).reshape(self.n_classes, self.n_classes)
        # Normalize along the rows
        self._confusion_matrix = self._confusion_matrix / self._confusion_matrix.sum(axis=1, keepdims=True)
        # Replace NaNs with 0
        self._confusion_matrix[np.isnan(self._confusion_matrix)] = 0
        # self.confusion_matrix += np.bincount(self.n_classes * target.reshape(-1) + pred.reshape(-1), minlength=self.n_classes**2).reshape(self.n_classes, self.n_classes)
        return self.value, self._confusion_matrix
    def plot_confusion_matrix(self, writer:SummaryWriter=None, global_step=None):
        """
        Plot the confusion matrix as an image.
        """
        fig,ax = plt.subplots()
        ax.imshow(self._confusion_matrix, cmap="Blues")
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                ax.text(j, i, round(self._confusion_matrix[i, j],2), ha="center", va="center", color="w")
        # Add faint line through the diagonal
        ax.plot([0, self.n_classes-1], [0, self.n_classes-1], color="k", linestyle="--",alpha=0.2)
        # ax.plot([0, self.n_classes-1], [0, self.n_classes-1], color="k", linestyle="--")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Target")
        ax.set_xticks(np.arange(self.n_classes), self.class_names[:self.n_classes])
        ax.set_yticks(np.arange(self.n_classes), self.class_names[:self.n_classes])
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")
        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        plt.show()
        if writer is not None:
            assert global_step is not None, "Global step must be provided when using a SummaryWriter."
            writer.add_figure("Confusion Matrix", fig, global_step)
    @property
    def value(self):
        return self.correct/self.count if self.count else 0
    @property
    def accuracy(self):
        return self.value
    @property
    def confusion_matrix(self):
        return self._confusion_matrix
    
class IoUMeter(Metric):
    def __init__(self, name="IoUMeter", n_classes:int=1):
        super(IoUMeter, self).__init__(name)
        self.n_classes = n_classes
        self.reset()
    def _reset(self):
        self.intersection = np.zeros(self.n_classes, dtype=np.int32)
        self.union = np.zeros(self.n_classes, dtype=np.int32)
    def _update(self, pred:Union[T.Tensor, np.ndarray], target:Union[T.Tensor, np.ndarray]):
        """
        Predictions and targets are expected to be in the same shape.
        The predicitons should contain the class index.
        Updating the metric will update the confusion matrix.
        """
        if isinstance(pred, T.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, T.Tensor):
            target = target.detach().cpu().numpy()
        assert pred.shape == target.shape, "Predictions and targets should have the same shape." # Shape should be (batch_size, H, W)
        for i in range(self.n_classes):
            self.intersection[i] += np.sum((pred == i) & (target == i))
            self.union[i] += np.sum((pred == i) | (target == i))
        return self.value
    @property
    def mean(self):
        self.m = np.mean(np.divide(self.intersection.astype(np.float32),self.union.astype(np.float32),where=self.union!=0,out=np.zeros_like(self.union,dtype=np.float32)))
        return self.m
    @property
    def value(self):
        return self.m if hasattr(self, "m") else self.mean
    @property
    def per_class(self):
        return np.divide(self.intersection.astype(np.float32),self.union.astype(np.float32),where=self.union!=0,out=np.zeros_like(self.union,dtype=np.float32))
    @property
    def class_i(self, i:int):
        return self.per_class[i]


class F1Meter(Metric):
    def __init__(self, name="F1Meter", n_classes:int=1):
        super(F1Meter, self).__init__(name)
        self.n_classes = n_classes
        self.reset()
    def _reset(self):
        self.tp = np.zeros(self.n_classes, dtype=np.int32)
        self.fp = np.zeros(self.n_classes, dtype=np.int32)
        self.fn = np.zeros(self.n_classes, dtype=np.int32)
    def _update(self, pred:Union[T.Tensor, np.ndarray], target:Union[T.Tensor, np.ndarray]):
        """
        Predictions and targets are expected to be in the same shape.
        The predicitons should contain the class index.
        Updating the metric will update the confusion matrix.
        """
        if isinstance(pred, T.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, T.Tensor):
            target = target.detach().cpu().numpy()
        assert pred.shape == target.shape, "Predictions and targets should have the same shape." # Shape should be (batch_size, H, W)
        for i in range(self.n_classes):
            self.tp[i] += np.sum((pred == i) & (target == i))
            self.fp[i] += np.sum((pred == i) & (target != i))
            self.fn[i] += np.sum((pred != i) & (target == i))
        return self.value
    @property
    def mean(self):
        s = self.tp + self.fp + self.fn
        # print(s, "HELLO")
        m = np.mean(np.divide(self.tp.astype(np.float32),s.astype(np.float32),where=s!=0,out=np.zeros_like(s,dtype=np.float32)))
        # Replace NaNs with 0
        self.m = m
        return m
    @property
    def value(self):
        return self.m if hasattr(self, "m") else self.mean
    @property
    def per_class(self):
        s = self.tp + self.fp + self.fn
        return np.divide(self.tp.astype(np.float32),s.astype(np.float32),where=s!=0,out=np.zeros_like(s,dtype=np.float32))
    @property
    def class_i(self, i:int):
        return self.per_class[i]

class MetricList:
    def __init__(self, name="MetricList", metrics:list[Metric]=None,n_classes:int=1):
        if len(metrics)>0 and isinstance(metrics[0], str):
            assert n_classes is not None, "n_classes must be provided when using a list of strings."
            self.metrics = [eval(metric)(n_classes) for metric in metrics]
        self.metrics = metrics
        self.initialized_plots = False
        self.updates = 0
    def update(self, *args, **kwargs):
        self.updates += 1
        for metric in self.metrics:
            metric.update(*args, **kwargs)
    def reset(self):
        for metric in self.metrics:
            metric.reset()
    def plot(self, writer=None, global_step=None):
        # Create subplot from returned figures
        if not self.initialized_plots:
            self.fig, self.axes = plt.subplots(1, len(self.metrics), figsize=(len(self.metrics)*5, 5))
        if not self.initialized_plots:
            self.lines = []
        for i, metric in enumerate(self.metrics):
            if not self.initialized_plots:
                self.axes[i].set_title(metric.name)
                self.lines.append(self.axes[i].plot([], [], label=metric.name)[0])
            metric.plot(metric.value,writer, global_step, ax=self.lines[i])
            # Relim and rescale axes
            self.axes[i].relim()
            self.axes[i].autoscale_view()
        if not self.initialized_plots:
            self.initialized_plots = True
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        if writer is not None:
            assert global_step is not None, "Global step must be provided when using a SummaryWriter."
            writer.add_figure("Metrics", self.fig, global_step)
    def to_dict(self):
        return {metric.name:metric.value for metric in self.metrics}
    def __getitem__(self, key):
        # Match key to metric name and return metric
        for metric in self.metrics:
            if metric.name == key:
                return metric
        raise KeyError(f"Metric with name {key} not found.")
    def summary(self):
        # Printable summary of metrics
        str_summary = ""
        for metric in self.metrics:
            str_summary += f"\n{metric.name}: {metric.value}"
        
        return str_summary
    def __len__(self):
        return len(self.metrics)
    def __repr__(self):
        return f"MetricList with {len(self.metrics)} metrics."
    def __str__(self):
        return self.summary().__str__()
    def __call__(self, *args: T.Tensor, **kwds: Any) -> Any:
        return self.update(*args, **kwds)
    @property
    def value(self):
        return sum(self.to_dict().values()) / len(self.metrics)
    

if __name__ == "__main__":
    np.random.seed(0)
    n_classes = 10
    size = 100
    batch_size = 10
    acc = AccuracyMeter(n_classes=n_classes)
    iou = IoUMeter(n_classes=n_classes)
    f1 = F1Meter(n_classes=n_classes)
    met_list = MetricList(metrics=[acc,iou,f1])
    for i in range(100):
        test_pred = T.tensor(np.random.randint(0,n_classes,(batch_size,1,size,size)))
        test_target = T.tensor(np.random.randint(0,n_classes,(batch_size,1,size,size)))
        met_list.update(test_pred,test_target)
        met_list.plot()
        plt.pause(0.1)
    plt.show()
