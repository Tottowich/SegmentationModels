import os
from datetime import datetime as dt
import logging
import numpy as np
import torch as T
import torch.nn as nn
import nvidia_smi
import psutil
from typing import Dict, List, Tuple, Union
from thop import clever_format, profile

try:
    from torchviz import make_dot
except ImportError:
    print("torchviz is not installed")


def view_vram(verbose:bool=False,percentage:bool=False)->dict:
    """
    Returns a dictionary with the total, used and free VRAM in MB, the keys are the device names.
    """
    nvidia_smi.nvmlInit()
    devices = {}
    assert nvidia_smi.nvmlDeviceGetCount() > 0, "No Nvidia-GPU found"
    for i in range(nvidia_smi.nvmlDeviceGetCount()):
        print(i)
        # Get device names
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        name = nvidia_smi.nvmlDeviceGetName(handle).decode("utf-8")
        if not percentage:
            devices[name] = {"total":info.total/1024/1024,
                            "used":info.used/1024/1024,
                            "free":info.free/1024/1024}
        else:
            devices[name] = {"total":100,
                            "used":info.used/info.total*100,
                            "free":info.free/info.total*100}
        if verbose:
            print(f"Device {name}:")
            print(f"Total: {devices[name]['total']}MB")
            print(f"Used: {devices[name]['used']}MB")
            print(f"Free: {devices[name]['free']}MB")
    return devices
def view_ram(verbose=False,percentage=False):
    """Returns the total, used and free RAM in MB"""
    mem = psutil.virtual_memory()
    if not percentage:
        usage = {"total":mem.total/1024/1024,
                "used":mem.used/1024/1024,
                "free":mem.free/1024/1024}
    else:
        usage = {"total":100,
                "used":mem.used/mem.total*100,
                "free":mem.free/mem.total*100}
    if verbose:
        print(f"Total: {usage['total']}MB")
        print(f"Used: {usage['used']}MB")
        print(f"Free: {usage['free']}MB")
    return usage
def cpu_usage(verbose=False):
    """Returns the total, used and free CPU in %"""
    cpu = psutil.cpu_percent()
    return cpu
def ram_usage():
    mem = psutil.virtual_memory()
    return mem.used/mem.total*100
def gpu_usage(verbose=False):
    """Returns the total, used and free GPU in %"""
    # Using nvidia_smi
    nvidia_smi.nvmlInit()
    assert nvidia_smi.nvmlDeviceGetCount() > 0, "No Nvidia-GPU found"
    usage = {}
    for i in range(nvidia_smi.nvmlDeviceGetCount()):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        # Get GPU name and utilization
        name = nvidia_smi.nvmlDeviceGetName(handle).decode("utf-8") 
        util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        if verbose:
            print(f"Device {name}:")
            print(f"Usage: {util.gpu}%")
            print(f"Memory: {util.memory}%")
        usage[name] = {"usage":util.gpu,
                        "memory":util.memory}
    return usage
def num_cpu():
    return psutil.cpu_count()
def num_gpu():
    nvidia_smi.nvmlInit()
    return nvidia_smi.nvmlDeviceGetCount()
def gpu_names():
    """Returns the names of the GPUs"""
    nvidia_smi.nvmlInit()
    assert nvidia_smi.nvmlDeviceGetCount() > 0, "No Nvidia-GPU found"
    names = []
    for i in range(nvidia_smi.nvmlDeviceGetCount()):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        name = nvidia_smi.nvmlDeviceGetName(handle).decode("utf-8") 
        names.append(name)
    return names
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

# def view_vram(*args, **kwargs):
#     nvidia_smi.nvmlInit()
#     for i in range(nvidia_smi.nvmlDeviceGetCount()):
#         handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
#         info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
#         print("Total: {}MB".format(info.total/1024/1024))
#         print("Used: {}MB".format(info.used/1024/1024))
#         print("Free: {}MB\n".format(info.free/1024/1024))
#     # Return the same arguments
#     return args, kwargs

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
        if self.visualize:
            os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
            self.plot_model(model,model.test_input)
        self.log_memory(model)
        self.log_parameters(model)
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
