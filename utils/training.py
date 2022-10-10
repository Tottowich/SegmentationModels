"""
Function and classes to information when training a model.
As well as functions to properly save checkpoints.
"""
import csv
import cv2
import datetime as dt
import os
import time
import copy
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
# WANDB imports:
import wandb
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from wandb.wandb_run import Run

from utils.helper import ModelLogger, normalize, to0_1, to_2array, to_2tuple
from utils.metrics import MetricList


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
    def __init__(self, model: nn.Module, optimizer: T.optim.Optimizer, criterion: nn.Module,
                 train_loader: DataLoader, val_loader: DataLoader,
                 test_loader: DataLoader, device: T.device, save_path: str,
                 hyper_parameters: Dict, wandb_run: Optional[Run] = None, epochs:int=100,log_interval: int = 10,
                 save_interval: int = 10, save_best: bool = True, save_last: bool = True,val_interval: int = 10,
                 checkpoint: Optional[str] = None, resume: bool = False, verbose: bool = True,metric_list:MetricList=None):
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
            epochs: Number of epochs to train for.
            log_interval: Number of batches between each log.
            save_interval: Number of batches between each save.
            save_best: If True, will save the best model.
            save_last: If True, will save the last model.
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
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
        self.epochs = epochs
        self.batch = 0
        self.start_time = time.time()
        self.end_time = time.time()
        self.val_interval = val_interval
        self.checkpoint = checkpoint
        self.resume = resume
        self.verbose = verbose
        self.metric_list = metric_list
        T.backends.cudnn.benchmark = True
        if self.checkpoint is not None:
            self.load_checkpoint(checkpoint)
        if not self.resume and self.checkpoint is not None:
            assert self.save_path != self.checkpoint, 'Save path and checkpoint path cannot be the same if a new training isnstance is to be created.'
            self.create_save_paths()
            self.create_dataset_info()
            self.create_model_info()
        elif not self.resume and self.checkpoint is None:
            self.create_save_paths()
            self.create_dataset_info()
            self.create_model_info()
            
    def create_save_paths(self):
        """
        Create all save paths.
        """
        if os.path.exists(self.save_path):
            # Add datetime to end of path.
            self.save_path = self.save_path + '_' + dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            print('Save path already exists, creating new save path: {}'.format(self.save_path))
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
        self.logger = ModelLogger(filename=os.path.join(self.model_info_dir, 'model_info.txt'),visualize=True)
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
            # Join image and label image together.
            # Label image is 1 channel, so we need to make it 3 channels.
            label = np.stack((label, label, label), axis=2)
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
        if self.resume:
            self.checkpoint_dir = os.path.join(self.save_path, 'checkpoints')
            self.plot_dir = os.path.join(self.save_path, 'plots')
            self.wandb_dir = os.path.join(self.save_path, 'wandb')
            self.dataset_info_dir = os.path.join(self.save_path, 'dataset_info')
            self.model_info_dir = os.path.join(self.save_path, 'model_info')
            self.example_batches_dir = os.path.join(self.save_path, 'example_batches')

    def load_checkpoint(self, path: str):
        """
        Load a checkpoint.
        Args:
            path: Path to the checkpoint.
        """
        checkpoint = T.load(path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epoch = checkpoint['epoch']
        self.batch = checkpoint['batch']
        self.best_loss = checkpoint['best_loss']
        self.best_acc = checkpoint['best_acc']
        self.best_epoch = checkpoint['best_epoch']
        self.train_loss = checkpoint['train_loss']
        self.val_loss = checkpoint['val_loss']
        self.train_acc = checkpoint['train_acc']
        self.val_acc = checkpoint['val_acc']
        self.train_time = checkpoint['train_time']
        self.val_time = checkpoint['val_time']
        self.hyper_parameters = checkpoint['hyper_parameters']
    def log_hyper_parameters(self):
        """
        Log hyper parameters to wandb.
        """
        hyp_path = os.path.join(self.save_path, 'hyper_parameters.yaml')
        # Add batch size to hyper parameters.
        self.hyper_parameters["batch_size"] = self.train_loader.batch_size
        with open(hyp_path, 'w') as f:
            yaml.dump(self.hyper_parameters, f)

        # wandb.config.update(self.hyper_parameters)
    def log_metrics(self):
        """
        Log metrics to csv file.
        """
        calculated_metrics = self.metric_list.summary() # Dictionary of metrics.
        metrics = {
            'epoch': self.epoch,
            'batch': self.batch,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'best_loss': self.best_loss,
            'best_acc': self.best_acc,
            'best_epoch': self.best_epoch,
            'best_loss_epoch': self.best_loss_epoch,
            'best_acc_epoch': self.best_acc_epoch
        }
        metrics.update(calculated_metrics)
        self.logger.info(metrics)
        # wandb.log(metrics)        
        with open(os.path.join(self.save_path, 'metrics.csv'), 'a') as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if self.epoch == 0 and self.batch == 0:
                writer.writeheader()
            writer.writerow(metrics)
        # Update WandB with metrics from calculated metrics.
        # wandb.log(calculated_metrics)


    def train(self):
        """
        Train the model.
        """
        # self.create_example_batches()
        self.logger.info('Starting training.')
        self.logger.timemark()
        self.logger.info(f'Epochs: {self.epochs}')
        self.logger.info(f'Batch size: {self.train_loader.batch_size}')
        self.logger.info(f'Number of training samples: {len(self.train_loader.dataset)}')
        self.logger.info(f'Number of training batches: {len(self.train_loader)}')
        self.logger.info(f'Number of validation batches: {len(self.val_loader)}')
        self.logger.info(f'Number of test batches: {len(self.test_loader)}')
        # self.logger.info(f'Number of classes: {self.num_classes}')

        # Start training !
        for epoch in range(self.epoch, self.epochs):
            self.epoch = epoch
            self.logger.epochmark(f"{epoch + 1}/{self.epochs}")
            self.train_epoch()
            if epoch % self.val_interval == 0:
                self.val_epoch() # Validate the model.
                # self.log_metrics() # Log metrics to csv file.
            if epoch % self.save_interval == 0:
                self.save_checkpoint(f'epoch_{epoch}', epoch, self.batch)
            if self.val_loss[-1] < self.best_loss:
                self.best_loss = self.val_loss[-1]
                self.best_loss_epoch = epoch
                save = True
                self.save_checkpoint(f'epoch_{epoch}', epoch, self.batch, best=True)
            if self.val_acc > self.best_acc:
                self.best_acc = self.val_acc
                self.best_acc_epoch = epoch
                self.save_checkpoint(f'epoch_{epoch}', epoch, self.batch, best=True)

        self.logger.info('Training complete.')
        self.logger.timemark()
        self.logger.log_model(self.model)
    def train_epoch(self):
        """
        Train the model for one epoch.
        """
        self.model.train()
        self.train_loss = 0
        for batch, (x, y) in enumerate(self.train_loader):
            self.batch += 1 # Increment batch counter for logging.
            x = x.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            y_pred,_,_ = self.model(x)
            loss = self.criterion(y_pred, y.float())
            loss.backward()
            self.optimizer.step()
            self.train_loss += loss.item()
        self.train_loss /= len(self.train_loader)
        self.logger.info(f'Train loss: {self.train_loss:.4f}')
        self.logger.timemark()
    def val_epoch(self):
        """
        Validate the model.
        """
        self.model.eval()
        val_loss = 0
        with T.no_grad():
            for batch, (x, y) in enumerate(self.val_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                y_pred,_,_ = self.model(x)
                loss = self.criterion(y_pred, y.float())
                val_loss += loss.item()
                y_pred = self.model.select_prediction(y_pred)
                y = self.model.select_prediction(y)
                self.metric_list.update(y_pred, y)
        val_loss /= len(self.val_loader)
        self.val_loss.append(val_loss)
        self.logger.info(f'Val loss: {val_loss:.4f} after {self.epoch + 1} epochs.')
        self.val_acc = self.metric_list.value
        self.logger.info(f'Val Metrics: {self.val_acc}')
        # self.logger.timemark()