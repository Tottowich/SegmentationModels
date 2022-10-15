import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.UNets import UNet
import torch as T   

# Create model
def create_model(config_file:str=None,checkpoint:str=None)->UNet:
    model = UNet(config=config_file,device="cuda" if T.cuda.is_available() else "cpu",checkpoint=checkpoint)
    return model