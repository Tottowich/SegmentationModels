import os
from datetime import datetime as dt
import logging
import numpy as np
import torch as T
import torch.nn as nn
import nvidia_smi
from thop import clever_format, profile

try:
    from torchviz import make_dot
except ImportError:
    print("torchviz is not installed")

def visualize_model(model,x):
    assert isinstance(model, T.nn.Module), "Model is not a torch.nn.Module"
    assert "make_dot" in globals(), "torchviz is not installed"
    
    if not isinstance(x,tuple):
        y,*_ = model(x)
    else:
        x,*context = x
        y,*_ = model(x,*context)
    g = make_dot(y.mean(), params=dict(model.named_parameters()))
    pdf_path = g.view(cleanup=True)
    return pdf_path

def to_2tuple(x):
    """Converts a number or iterable to a 2-tuple."""
    if isinstance(x, (list, tuple)):
        return x
    return (x, x)
def to_2array(x):
    """Converts a number or iterable to a 2-array."""
    if isinstance(x, (list, tuple)):
        return np.array(x)
    return np.array([x, x])

def normalize(x, eps=1e-8):
    """Normalizes a tensor to have unit norm."""
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def to0_1(x):
    """Converts a tensor to [0,1]"""
    return (x - x.min()) / (x.max() - x.min())

def view_vram(*args, **kwargs):
    nvidia_smi.nvmlInit()
    for i in range(nvidia_smi.nvmlDeviceGetCount()):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print("Total: {}MB".format(info.total/1024/1024))
        print("Used: {}MB".format(info.used/1024/1024))
        print("Free: {}MB\n".format(info.free/1024/1024))
    # Return the same arguments
    return args, kwargs

class ModelLogger(logging.Logger):
    """
    When calling verbose on a model, it will log the model's parameters and buffers, shapes.
    Upon initialization the model will also log:
        - the number of parameters and buffers
        - the number of flops and parameters
        - the number of bytes of memory
    """
    def __init__(self, model:nn.Module=None, level=logging.DEBUG,filename:str=None,visualize:bool=False) -> None:
        name = model.__class__.__name__ if model is not None else "ModelLogger"
        super().__init__(name, level)
        self.setLevel(level)
        if filename is None:
            self.handler = logging.StreamHandler()
        else:
            self.handler = logging.FileHandler(filename)
        self.handler.setLevel(level)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.handler.setFormatter(self.formatter)
        self.addHandler(self.handler)
        self.visualize = visualize
        if model is not None:
            self.log_model(model)
    def log_model(self,model):
        self.log_memory(model)
        self.log_parameters(model)
        if self.visualize:
            os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
            self.plot_model(model,model.test_input)
    def log_parameters(self,model):
        self.flop_counter(model,model.test_input)
    def log_memory(self,model):
        self.memory_usage(model)
    def plot_model(self,model,x):
        if not isinstance(x,tuple):
            y,*_ = model(x)
        else:
            x,*context = x
            y,*_ = model(x,*context)
        g = make_dot(y.mean(), params=dict(model.named_parameters()))
        pdf_path = g.view(cleanup=True)
        return pdf_path

    def flop_counter(self,model,x,**kwargs):
        if not isinstance(x,tuple):
            flops, params = profile(model, inputs=(x,),verbose=False)
        else:
            x,*context = x
            flops, params = profile(model, inputs=(x,*context),verbose=False)
            del context
        self.info(f"{model.__class__.__name__} uses {flops/1e9:.2f} GFLOPS and {params/1e6:.2f} MParams")
        del x 
    def memory_usage(self,model):
        mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
        mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
        mem = mem_params + mem_bufs # in bytes
        self.info(f"{model.__class__.__name__} has {mem/1e6:.2f} MBytes of memory")
    def timestamp(self):
        return dt.now().strftime("%Y-%m-%d %H:%M:%S")
    def timemark(self):
        self.info(f"{'='*20} {self.timestamp()} {'='*20}")
    def epochmark(self,epoch):
        self.info(f"{'='*20} Epoch {epoch} staring at {self.timestamp()}{'='*20}")
    def blank(self):
        # Black line without formatting
        self.info("")
