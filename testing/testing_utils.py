# Useful functions for testing the different parts of the project
import os
import sys
import torch as T
import traceback
from colorama import Fore, Back, Style
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
# import timeit
import time
from thop import profile, clever_format
try:
    from torchviz import make_dot
except ImportError:
    print("torchviz is not installed")
import logging


def check_backends():
    if T.cuda.is_available():
        print("CUDA is available")
        device = T.device("cuda")
    elif T.backends.mps.is_available():
        print("MPS is available")
        device = T.device("cpu")
    else:
        print("CUDA nor MPS is not available")
        device = T.device("cpu")
    return device




# Timing decorator
def time_func(method):
    def timed(*args, **kw):
        ts = time.monotonic()
        result = method(*args, **kw)
        te = time.monotonic()
        print(f"{method.__name__} took {te-ts:.6e} seconds")
        return result
    return timed
# # Flops decorator
# def flops_func(method):
#     def flops(*args, **kw):
#         model = method(*args, **kw)
#         assert isinstance(model, T.nn.Module), "Model is not a torch.nn.Module"
#         flops, params = profile(model, inputs=(T.randn(1,3,128,128).cuda(),))
#         flops, params = clever_format([flops, params], "%.3f")
#         # print(f"{method.__name__} has {flops/1e9:.2f} GFLOPS and {params/1e6:.2f} MParams")
#         print(f"{method.__name__} has {flops} GFLOPS and {params} MParams")
#         return model
#     return flops
# Test backpropagation
def test_backprop(model, x):
    if isinstance(x,tuple):
        x,*context = x
        y,*_ = model(x,*context)
    else:
        y,*_ = model(x)
    loss = T.mean(y)
    loss.backward()
    print(f"Backpropagation test passed for {model.__class__.__name__}")
    del loss
    del y

def flop_counter(model,x,**kwargs):
    # if kwargs.get("context",None) is None:
    #     flops, params = profile(model, inputs=(x,),verbose=False)
    # else:
    #     flops, params = profile(model, inputs=(x,kwargs.get("context",None)),verbose=False)
    if not isinstance(x,tuple):
        flops, params = profile(model, inputs=(x,),verbose=False)
    else:
        x,*context = x
        flops, params = profile(model, inputs=(x,*context),verbose=False)
        del context
    print(f"{model.__class__.__name__} has {flops/1e9:.2f} GFLOPS and {params/1e6:.2f} MParams")
    del x 
def memory_usage(model):
    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs # in bytes
    print(f"{model.__class__.__name__} has {mem/1e6:.2f} MBytes of memory")

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

# def pdf2png(pdf_path):
#     png_path = pdf_path.replace(".pdf",".png")
#     imgkit.from_file(pdf_path,png_path)
#     return png_path

class TestLogger(logging.Logger):
    def __init__(self, name, level=logging.DEBUG,filename=None):
        super().__init__(name, level)
        self.setLevel(level)
        if filename is None:
            self.handler = logging.StreamHandler()
        else:
            self.handler = logging.FileHandler(filename)
        # self.handler.setLevel(level)
        # self.handler = logging.StreamHandler()
        self.handler.setLevel(level)
        self.formatter = logging.Formatter('%(asctime)s :: %(name)s - %(levelname)s - %(message)s')
        self.handler.setFormatter(self.formatter)
        self.addHandler(self.handler)


    # def __del__(self):
    #     self.removeHandler(self.handler)
    #     self.handler.close()
    def __call__(self, msg):
        self.info(msg)
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.removeHandler(self.handler)
        self.handler.close()
    def initialize_test(self,method):
        self.info(Fore.LIGHTMAGENTA_EX + f"Starting unit test for | {method} |" + Style.RESET_ALL)
    def completed_test(self,method):
        self.info(Fore.GREEN + f"Unit test for | {method} | passed\n" + Style.RESET_ALL)
    def failed_test(self,method):
        self.warning(Fore.RED + f"Unit test for | {method} | failed ->" + Style.RESET_ALL)
        print("-"*60)
        print(traceback.format_exc(),end="")
        print("-"*60+"\n")


