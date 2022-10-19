"""
Function and classes to information when training a model.
As well as functions to properly save checkpoints.
"""
import csv
import cv2
import datetime as dt
import os
import sys
import time
import copy
from typing import Dict, List, Optional, Tuple, Union, Type
import matplotlib.pyplot as plt
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
# WANDB imports:
import wandb
import yaml
import inspect
from PIL import Image
from torch.utils.data import DataLoader
from wandb.wandb_run import Run
from tqdm import tqdm
from models.UNets import UNet
if __name__=="__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helper import view_vram, view_ram, gpu_usage,cpu_usage,ram_usage, gpu_names
from utils.helper import ModelLogger#, normalize, to0_1, to_2array, to_2tuple
from loss.loss_functions import UNetLossFunction
from torch import optim
from data.loaders import get_dataloader, ADE20K, ADE20KSingleExample, split_dataset
from torch.utils.data import random_split
from utils.visualizations import SegmentationVisualizer
from utils.metrics import MetricList
EPS = 1e-6
class ProgBar(tqdm):
    def __init__(self, duration:int=100,tracking:str=["gpu","cpu"],*args, **kwargs):
        """
        A progress bar that can track the system resources.
        Tracking:
            - gpu: GPU memory and usage.
            - cpu: CPU memory and usage.
        """
        # Set bar format with length 10 of the bar.
        super().__init__(*args, **kwargs, total=duration,leave=True)
        self.tracking = tracking
        self._duration = duration
        self.bar_format = '{desc}{percentage:3.0f}%|{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        # Set tqdm format.

        # self.format_meter("{desc}: {percentage:3.0f}%|{bar:<10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")
        if "cpu" in self.tracking:
            self.metrics = {"cpu":{},"gpu":{}}
            self.metrics["cpu"]["memory"] = []
            self.metrics["cpu"]["usage"] = []
        # self._device_names = gpu_names()
        if "gpu" in self.tracking:
            for i in range(10):
                for name in gpu_names():
                    self.metrics["gpu"][name] = {}
                    self.metrics["gpu"][name]["memory"] = []
                    self.metrics["gpu"][name]["usage"] = []

    def step(self, n: float, loss: Optional[float] = None, train: bool = True):
        """
        Update the progress bar.
        """
        if self._duration == -1:
            self.total = n
        super().update(n)
        self.update_stats()
        if loss is not None:
            self.display_loss(loss,train)
    def update_stats(self):
        """
        Update the stats.
        """
        stats = {}
        desc = ""
        if "gpu" in self.tracking:
            gpu = gpu_usage()
            for k,v in gpu.items():
                desc += f"| Device: {k} | VRAM: {v['memory']:.2f}% | GPU: {v['usage']:.2f}% |"
                self.metrics["gpu"][k]["usage"].append(v["usage"])
                self.metrics["gpu"][k]["memory"].append(v["memory"])
            # desc += f"| VRAM: {gpu['memory']:.2f}MB |"
            # desc += f"| GPU: {gpu['usage']:.2f}% |"
        if "cpu" in self.tracking:
            cpu = cpu_usage()
            ram = ram_usage()
            desc += "" if "gpu" in self.tracking else "|"
            desc += f" RAM: {ram:.2f}% | CPU: {cpu:.2f}% |"
            self.metrics["cpu"]["usage"].append(cpu)
            self.metrics["cpu"]["memory"].append(ram)
        # return desc
        self.set_description(desc)
    def close(self):
        """
        Close the progress bar.
        """
        # self.update_stats()
        super().close()
    def plot(self,show:bool=True,savepath:str=None):
        """
        Plot the metrics.
        """
        # Metrics has structure:
        # metrics = {"cpu":{"memory":[],"usage":[]},"gpu":{device_names}:{"memory":[],"usage":[]}}
        # Create Subplots for each metric.
        fig,ax = plt.subplots(nrows=2,ncols=2,figsize=(20,9))
        # fig,ax = plt.subplots(nrows=len(self.metrics["gpu"].keys())+1,ncols=2,figsize=(12,9))
        # Plot CPU metrics.
        ax[0,0].plot(self.metrics["cpu"]["memory"],label="RAM")
        ax[0,0].set_title("CPU RAM Usage")
        ax[0,0].set_xlabel("Epoch")
        ax[0,0].set_ylabel("RAM Usage (%)")
        # ax[0,0].lengend()
        ax[0,1].plot(self.metrics["cpu"]["usage"],label="CPU")
        ax[0,1].set_title("CPU Usage")
        ax[0,1].set_xlabel("Epoch")
        ax[0,1].set_ylabel("CPU Usage (%)")
        # ax[0,1].legend()
        # Plot GPU metrics.
        for i,(k,v) in enumerate(self.metrics["gpu"].items()):
            ax[1,0].plot(v["memory"],label=k)
            ax[1,0].set_title("GPU VRAM Usage")
            ax[1,0].set_xlabel("Epoch")
            ax[1,0].set_ylabel("VRAM Usage (%)")
            # Set legend location to the right.
            ax[1,0].legend(loc="center left",bbox_to_anchor=(1,0.5))
            # ax[i+1,0].legend()
            ax[1,1].plot(v["usage"],label=k)
            ax[1,1].set_title("GPU Usage")
            ax[1,1].set_xlabel("Epoch")
            ax[1,1].set_ylabel("GPU Usage (%)")
            ax[1,1].legend(loc="center left",bbox_to_anchor=(1,0.5))
            # ax[i+1,1].legend()
        # Create subplot of all 
        # ax.legend()
    
        fig.legend(loc="center left",bbox_to_anchor=(1,0.5))
        fig.tight_layout()
        if savepath is not None:
            plt.savefig(savepath)
        if show:
            plt.show()
    def display_loss(self,t_loss:float,train=True):
        """
        Display the loss.
        """
        msg = f"Train Loss: {t_loss:.4f}" if train else f"Val Loss: {t_loss:.4f}"
        self.set_postfix_str(msg)
        # self.step(0)

        
def load_model(model, path, device):
    checkpoint = T.load(path)
    config = checkpoint['model_config']
    print('Loading model from checkpoint: {}'.format(path))
    model = model(config) # When loading a previous
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    return model,checkpoint


class Trainer:
    """
    Class to train model. Will log the training process to Weights and Biases.
    The trainer handels the logging of information to local folder as well.
    This folder will include:
        - Hyperparameters: Yaml file containing hyper parameters used when training.
        TODO: - Weights and Biases: Weights and Biases project.
        TODO: - Tensorboard: Tensorboard project.
        TODO: - Learningrate scheduler: Learningrate scheduler.

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
                 checkpoint: Optional[str] = None, resume: bool = False, verbose: bool = True,metric_list:MetricList=None,pbar:Union[ProgBar,bool]=True,n_classes:int=151,class_names:list[str]=None):
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
        self.checkpoint = checkpoint
        self.model = model # Model could be uninitialized. If so the input parameter should be a function that returns a model given a dictionary config created by that model.
        self.optimizer = optimizer
        self.verbose = verbose
        self.resume = resume
        self.device = device
        if self.checkpoint is not None:
            self.load_checkpoint(checkpoint)
        else: 
            self.epoch = 0
            self.batch = 0
            self.best_loss = np.inf
            self.best_acc = 0
            self.best_epoch = 0
            self.train_loss = []
            self.val_loss = []
            self.train_acc = []
            self.val_acc = []
            self.train_time = []
            self.val_time = []
            self.hyper_parameters = hyper_parameters
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.save_path = save_path
        self.wandb_run = wandb_run
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.save_best = save_best
        self.save_last = save_last
        self.best_model = None
        self.epochs = epochs
        self.start_time = time.time()
        self.end_time = time.time()
        self.val_interval = val_interval
        self.metric_list = metric_list
        self.pbar = pbar
        self.criterion = criterion
        self.n_classes = n_classes
        self.class_names = class_names
        self.seg_visualizer = SegmentationVisualizer(n_classes=n_classes,class_names=class_names)
        self.model = self.model.to(self.device)
        if self.val_loader is None:
            self.val_loader = self.train_loader
        if self.test_loader is None:
            self.test_loader = self.val_loader
        T.backends.cudnn.benchmark = True
        self.create_save_paths()
        self.create_dataset_info()
        self.create_model_info()
        self.create_example_batches()
        print("Trainer initialized.")
        if not self.resume:
            # assert self.save_path != self.checkpoint, 'Save path and checkpoint path cannot be the same if a new training isnstance is to be created.'
            # self.create_dataset_info()
            pass
            
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
        self.logger.info(self.optimizer) # Print optimizer to file.
        # Get all hyper parameters from optimizer.

        self.logger.timemark()
    def create_dataset_info(self):
        """
        Create dataset info.
        """
        # TODO: Create dataset analysis
        pass
    def create_example_batches(self,train:bool=True):
        """
        Create example batches.
        Get an example batch from the train loader and save it to a folder visually.
        """
        if train:
            loader = self.train_loader
        else:
            loader = self.val_loader
        for trainx, trainy in loader:
            trainy = self.model.select_prediction(trainy)
            fig = self.seg_visualizer.grid(trainx.numpy(),trainy.numpy(),None)
            break
        for txt in fig.texts:
            txt.set_visible(False)
        # Get axes from figure.
        axes = fig.get_axes()
        for ax in axes:
            for txt in ax.texts:
                txt.set_visible(False)
        fig.savefig(os.path.join(self.example_batches_dir, 'example_batch.png'))

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
            'model_config': self.model.config,
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
        # checkpoint = T.load(path)
        # config = checkpoint['model_config']
        # print('Loading model from checkpoint: {}'.format(path))
        # self.model = self.model(config) # When loading a previous
        # self.model.load_state_dict(checkpoint['model'])
        assert os.path.exists(path), 'Checkpoint does not exist: {}'.format(path)
        assert path.endswith('.pt') or path.endswith('.pth'), 'Checkpoint must be a .pt or .pth file.'
        checkpoint = T.load(path)
        self.hyper_parameters = checkpoint['hyper_parameters']
        self.optimizer = self.optimizer(self.model.parameters(), **self.hyper_parameters)
        if self.resume:
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
        else:
            self.epoch = 0
            self.batch = 0
            self.best_loss = np.inf
            self.best_acc = 0
            self.best_epoch = 0
            self.train_loss = []
            self.val_loss = []
            self.train_acc = []
            self.val_acc = []
            self.train_time = []
            self.val_time = []


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
        calculated_metrics = self.metric_list.summary() if self.metric_list is not None else None# Dictionary of metrics.
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
        if calculated_metrics is not None:
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
        self.logger.info(f"Number of validation samples: {len(self.val_loader.dataset)}")
        self.logger.info(f'Number of test batches: {len(self.test_loader)}')
        self.logger.info(f"Number of test samples: {len(self.test_loader.dataset)}")

        if self.pbar:
            self.epoch_pbar = ProgBar(self.epochs,position=0)
            self.epoch_pbar.step(self.epoch, loss=self.val_loss[-1] if len(self.val_loss) > 0 else 0)
            self.batch_pbar = ProgBar(len(self.train_loader),position=1)
        for epoch in range(self.epoch, self.epochs):
            self.epoch = epoch
            self.logger.epochmark(f"{epoch + 1}/{self.epochs}")
            if self.pbar:
                self.epoch_pbar.step(1,loss=self.val_loss[-1] if self.val_loss else 0,train=False)
                self.batch_pbar.reset()
            # tic = time.time()
            self.train_epoch()
            # toc = time.time()
            # print(f"Epoch {epoch} took {toc-tic} seconds")
            if epoch+1 % self.val_interval == 0 or epoch == 0:
                self.val_epoch() # Validate the model.
                # self.log_metrics() # Log metrics to csv file.
                if self.val_loss[-1] < self.best_loss:
                    self.best_loss = self.val_loss[-1]
                    self.best_loss_epoch = epoch
                    self.save_checkpoint(f'epoch_{epoch}', epoch, self.batch, best=True)
            if epoch % self.save_interval == 0:
                self.save_checkpoint(f'epoch_{epoch}', epoch, self.batch)
            # if self.val_acc > self.best_acc:
            #     self.best_acc = self.val_acc
            #     self.best_acc_epoch = epoch
            #     self.save_checkpoint(f'epoch_{epoch}', epoch, self.batch, best=True)

        self.logger.info('Training complete.')
        self.logger.timemark()
        self.logger.log_model(self.model)
    def train_epoch(self):
        """
        Train the model for one epoch.
        """
        self.model.train()
        train_loss = 0
        tic1 = time.time()
        for batch, (x, y) in enumerate(self.train_loader):
            # toc1 = time.time()
            # print(f"Loading batch {batch} took {toc1-tic1} seconds")
            self.batch += 1 # Increment batch counter for logging.
            # tic = time.time()
            x = x.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            y_pred,_,_ = self.model(x)
            loss = self.criterion(y_pred, y.float())
            loss.backward()
            self.optimizer.step()
            # toc = time.time()
            # print(f"Batch {batch} took {toc-tic} seconds")
            train_loss += loss.item()
            if self.pbar:
                self.batch_pbar.step(1,loss.item())
            # tic1 = time.time()
        self.train_loss.append(train_loss/len(self.train_loader))
        self.logger.info(f'Train loss: {self.train_loss[-1]:.4f}')
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
        if len(self.metric_list):
            self.val_acc = self.metric_list.value
            self.logger.info(f'Val Metrics: {self.val_acc}')
        # self.logger.timemark()
# Create model
def create_model(config_file:str=None,checkpoint:str=None)->Type[UNet]:
    model = UNet(config=config_file,device="cuda" if T.cuda.is_available() else "cpu",checkpoint=checkpoint)
    return model

# Create optimizer
def create_optimizer(opt:str="SGD",model=None,hyps:dict=None)->Type[optim.Adam]:
    optimizers = {
        "SGD":optim.SGD,
        "Adam":optim.Adam,
        "AdamW":optim.AdamW,
        "Adadelta":optim.Adadelta,
        "Adagrad":optim.Adagrad,
        "Adamax":optim.Adamax,
        "ASGD":optim.ASGD,
        "LBFGS":optim.LBFGS,
        "RMSprop":optim.RMSprop,
        "Rprop":optim.Rprop,
        "SparseAdam":optim.SparseAdam,
    }
    assert opt in optimizers.keys(), f"Optimizer {opt} not supported only supports {optimizers.keys()}"
    # Check that keys in hyps are valid for optimizer
    valid_keys = inspect.getfullargspec(optimizers[opt]).args
    valid_hyps = dict([(k,v) for k,v in hyps.items() if k in valid_keys])
    if model is None:
        return optimizers[opt]
    else:
        return optimizers[opt](model.parameters(),**valid_hyps)

# Create loss function
def create_criterion(alpha:float=1,beta:float=1,smooth:float=1,eps:float=1e-8)->Type[UNetLossFunction]:
    loss_function = UNetLossFunction(alpha=alpha,beta=beta,smooth=smooth,eps=eps)
    return loss_function

# Create dataloader
def create_ADE20K_dataset(img_size,cache,fraction=1.0,train:bool=True,transform=None,single_example=False,index=0)->Union[ADE20K,ADE20KSingleExample]:
    if isinstance(img_size,(tuple,list)):
        img_size = img_size[0] # TODO: Make this more general for rectangular images
    if single_example:
        dataset = ADE20KSingleExample(img_size=img_size,transform=transform,fraction=fraction,categorical=True,index=index,train=train)
    else:
        dataset = ADE20K(cache=cache,img_size=img_size,fraction=fraction,transform=transform,categorical=True,train=train)
    return dataset

# Create dataloader
def create_dataloader(dataset,batch_size,num_workers):
    dataloader = get_dataloader(dataset,batch_size=batch_size,num_workers=num_workers)
    return dataloader

# Create trainer
def create_trainer(
                model,
                optimizer,
                criterion,
                batch_size,
                num_workers,
                device,
                save_path,
                hyper_parameters,
                wandb_run,
                epochs,
                log_interval,
                save_interval,
                save_best,
                save_last,
                val_interval,
                checkpoint,
                resume,
                verbose,
                metric_list,
                pbar,
                train_loader:Type[DataLoader]=None,
                val_loader:Type[DataLoader]=None,
                test_loader:Type[DataLoader]=None,
                dataset:Union[ADE20K,ADE20KSingleExample]=None,
                )->Type[Trainer]:
    if dataset is not None and len(dataset) > 1:
        train_set, val_set, test_set = split_dataset(dataset)
        train_loader = create_dataloader(train_set,batch_size,num_workers)
        val_loader = create_dataloader(val_set,batch_size,num_workers)
        test_loader = create_dataloader(test_set,batch_size,num_workers)
    elif dataset is not None:
        train_loader = create_dataloader(dataset,batch_size,num_workers)
        val_loader = train_loader
        test_loader = train_loader
    else:
        assert train_loader is not None, "Must provide train_loader if dataset is None"
        assert val_loader is not None, "Must provide val_loader if dataset is None"
        assert test_loader is not None, "Must provide test_loader if dataset is None"

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        save_path=save_path,
        hyper_parameters=hyper_parameters,
        wandb_run=wandb_run,
        epochs=epochs,
        log_interval=log_interval,
        save_interval=save_interval,
        save_best=save_best,
        save_last=save_last,
        val_interval=val_interval,
        checkpoint=checkpoint,
        resume=resume,
        verbose=verbose,
        metric_list=metric_list,
        pbar=pbar,
        n_classes=dataset.num_classes,
        class_names=dataset.class_names,
    )
    return trainer