"""
Function and classes to information when training a model.
As well as functions to properly save checkpoints.
"""
import os
import time
import datetime as dt
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
# WANDB imports:
import wandb
from wandb.wandb_run import Run

from utils.helper import ModelLogger, normalize, to0_1, to_2array, to_2tuple


class Trainer:
    """
    Class to train model. Will log the training process to Weights and Biases.
    The trainer handels the logging of information to local folder as well.
    This folder will include:
        - Hyperparameters: Yaml file containing hyper parameters used when training.
        - ModelInfo: Folder containing a text file discribing the model, as well as a plot of the model.
        - Dataset information: Yaml file containing differnt type of information regarding the used dataset;
            - Distrubution of classes,
            - Size of dataset.
            - Regional distrubution of classes.
            - Metrics regarding each class.
                - Average coverage of a class.
                TODO: Find more interesting metrics regarding the dataset used.
        - Example batches: Folder of images containing samples of images used when training.
        - Checkpoints: Storing at an interval, best and last model. The optimizer state should also be stored to
                       make resumed training possible.
        - Logs: txt file, All logs of the training session should be stored in a file.
        - WandB: Folder containing relevant WandB Information.
        - Plots: Folder containing plots of the training process.
    """
    def __init__(self, model: nn.Module, optimizer: T.optim.Optimizer, loss_function: nn.Module,
                 train_loader: T.utils.data.DataLoader, val_loader: T.utils.data.DataLoader,
                 test_loader: T.utils.data.DataLoader, device: T.device, save_path: str,
                 hyper_parameters: Dict, wandb_run: Optional[Run] = None, log_interval: int = 10,
                 save_interval: int = 10, save_best: bool = True, save_last: bool = True):
        """
        Args:
            model: Model to train.
            optimizer: Optimizer to use when training.
            loss_function: Loss function to use when training.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            test_loader: DataLoader for test data.
            device: Device to train on.
            save_path: Path to save checkpoints and logs.
            hyper_parameters: Hyper parameters used when training.
            wandb_run: Run to log information to.
            log_interval: Number of batches between each log.
            save_interval: Number of batches between each save.
            save_best: If True, will save the best model.
            save_last: If True, will save the last model.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.model = model.to(self.device)
        self.save_path = save_path
        self.hyper_parameters = hyper_parameters
        self.wandb_run = wandb_run
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.save_best = save_best
        self.save_last = save_last
        self.best_loss = np.inf
        self.best_acc = 0
        self.best_epoch = 0
        self.best_model = None
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.train_time = []
        self.val_time = []
        self.epoch = 0
        self.batch = 0
        self.start_time = time.time()
        self.end_time = time.time()
        self.create_save_paths()
        self.create_dataset_info()
        self.create_model_info()
    def create_save_paths(self):
        """
        Create all save paths.
        """
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'checkpoints'), exist_ok=True)
        self.checkpoint_dir = os.path.join(self.save_path, 'checkpoints')
        os.makedirs(os.path.join(self.save_path, 'plots'), exist_ok=True)
        self.plot_dir = os.path.join(self.save_path, 'plots')
        os.makedirs(os.path.join(self.save_path, 'wandb'), exist_ok=True)
        self.wandb_dir = os.path.join(self.save_path, 'wandb')
        os.makedirs(os.path.join(self.save_path, 'dataset_info'), exist_ok=True)
        self.dataset_info_dir = os.path.join(self.save_path, 'dataset_info')
        os.makedirs(os.path.join(self.save_path, 'model_info'), exist_ok=True)
        self.model_info_dir = os.path.join(self.save_path, 'model_info')
        os.makedirs(os.path.join(self.save_path, 'example_batches'), exist_ok=True)
        self.example_batches_dir = os.path.join(self.save_path, 'example_batches')
    def create_model_info(self):
        """
        Create model info.
        """
        self.logger = ModelLogger(self.model, filename=os.path.join(self.model_info_dir, 'model_info.txt'),visualize=True)
        # Model is saved as PDF for now: TODO: Change to PNG or JPG if possible.
        self.logger.info(self.model) # Print model to file.
        self.logger.timemark()
    def create_dataset_info(self):
        """
        Create dataset info.
        """
        # TODO: Create dataset analysis
        pass
    def create_example_batches(self):
        """
        Create example batches.
        Get an example batch from the train loader and save it to a folder visually.
        """
        trainx, trainy = next(iter(self.train_loader)) # Get first batch.
        # Store trainx and trainy as a side by side image.
        for i in range(len(trainx)):
            img = trainx[i]
            label = trainy[i]
            img = img.permute(1, 2, 0) # Change to HWC
            img = img.cpu().numpy()
            img = img * 255
            img = img.astype(np.uint8)
            label = label.cpu().numpy()
            label = label.astype(np.uint8)
            combined  = np.concatenate((img, label), axis=1)
            combined = Image.fromarray(combined)
            combined.save(os.path.join(self.example_batches_dir, f'example_batch_{i}.png'))
    # def create_logs(self):
            # img = Image.fromarray(img)
            # label = Image.fromarray(label)
            # Create side by side image.



    def save_checkpoint(self, name: str, epoch: int, batch: int, best: bool = False):
        """
        Save a checkpoint.
        Args:
            name: Name of the checkpoint.
            epoch: Epoch of the checkpoint.
            batch: Batch of the checkpoint.
            best: If True, will save the best model.
        """
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
            'batch': batch,
            'best_loss': self.best_loss,
            'best_acc': self.best_acc,
            'best_epoch': self.best_epoch,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'train_acc': self.train_acc,
            'val_acc': self.val_acc,
            'train_time': self.train_time,
            'val_time': self.val_time,
            'hyper_parameters': self.hyper_parameters
        }
        if best:
            T.save(checkpoint, os.path.join(self.save_path, 'checkpoints', 'best_model.pt'))
        else:
            T.save(checkpoint, os.path.join(self.save_path, 'checkpoints', f'{name}.pt'))

        