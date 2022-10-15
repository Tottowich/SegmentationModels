# Functions and classes used to improve optimizer performance and ease of use
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np

# Optimizer class that includes gradient clipping, learning rate scheduling and gradient accumulation
class Optimizer:
    def __init__(self,optimizer,clip_grad=None,lr_scheduler:lr_scheduler._LRScheduler=None,weight_decay_scheduler=None,momentum_scheduler=None,gradient_accumulation_steps=1):
        self.optimizer = optimizer
        self.clip_grad = clip_grad
        self.lr_scheduler = lr_scheduler
        self.weight_decay_scheduler = weight_decay_scheduler
        self.momentum_scheduler = momentum_scheduler
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.iteration = 0

    def step(self):
        if self.gradient_accumulation_steps%self.iteration == 0:
            if self.clip_grad is not None:
                nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'],self.clip_grad)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self.weight_decay_scheduler is not None:
                self.weight_decay_scheduler.step()
            if self.momentum_scheduler is not None:
                self.momentum_scheduler.step()
        self.iteration += 1

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self,state_dict):
        self.optimizer.load_state_dict(state_dict)

    def __getattr__(self,name):
        return getattr(self.optimizer,name)