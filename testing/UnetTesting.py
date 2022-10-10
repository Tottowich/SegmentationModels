import os
from re import I
import sys
from collections import namedtuple
from tkinter import HORIZONTAL
from tkinter.tix import Tree
from colorama import Fore, Back, Style
from PIL import Image
from copy import copy
from torch.cuda import empty_cache
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.UNets import *
from models.Attention import *
from testing_utils import visualize_model
from testing_utils import time_func,test_backprop, flop_counter, memory_usage, TestLogger, check_backends
import inspect
from inspect import getmembers, isfunction
import PySimpleGUI as sg
import traceback
# Test the various classes in the UNets.py file
WIDTH = 512
HEIGHT = 512
IN_CHANNELS = 3
OUT_CHANNELS = 32
BATCH_SIZE=1
EMBED_DIM = 768
INPUT_SHAPE = (BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)
PARAMS = {
        "width":WIDTH,
        "height":HEIGHT,
        "encoder_blocks":4,
        "encoder_block_depth":3,   
        "kernel_size":3,
        "in_channels":IN_CHANNELS, 
        "out_channels":OUT_CHANNELS, 
        "batch_size":BATCH_SIZE, 
        "embed_dim":EMBED_DIM,
        "num_heads":8,
        "patch_size":16,}
SCRIPT_NAME = os.path.basename(__file__).split(".")[0]


class Tester:
    """
    Tester contains all testing functions.
    Used to have GUI for selecting method to test.
    All functions named test_* are automatically added to the GUI.
    """
    def __init__(self,params:dict):
        device = check_backends()
        self.params = params
        self.width = int(params["width"])
        self.height = int(params["height"])
        self.in_channels = int(params["in_channels"])
        self.out_channels = int(params["out_channels"])
        self.batch_size = int(params["batch_size"])
        self.embed_dim = int(params["embed_dim"])
        self.num_heads = int(params["num_heads"])
        self.patch_size = int(params["patch_size"])
        self.encoder_blocks = int(params["encoder_blocks"])
        self.encoder_block_depth = int(params["encoder_block_depth"])

        self.input_shape = (int(self.batch_size), int(self.in_channels), int(self.height), int(self.width))
        self.ex_img = T.randn(*self.input_shape).to(device)
        self.ex_mask = T.randn(*self.input_shape).to(device)
        self.patch_stride = self.patch_size
        self.num_patches = (self.width // self.patch_stride) * (self.height // self.patch_stride)
        self.patch_shape = (self.batch_size, self.num_patches, self.embed_dim)
        self.patched_img = T.randn(*self.patch_shape).to(device)
        self.patched_mask = T.randn(*self.patch_shape).to(device)
        self.device = device
        # print(f"Testing with: {self.params}")
        # Get name of all methods in this class
        self.methods = ["_".join(o[0].split("_")[1:]) for o in getmembers(Tester, predicate=isfunction) if not o[0].startswith("__") and o[0].startswith("test_")]
        
    def unit_test_complete(self,class_name:str):
        print(Fore.GREEN + f"Unit test for | {class_name} | passed\n\n" + Style.RESET_ALL)

    @time_func
    def test_ViTAttention(self):
        img = self.ex_img
        x = self.patched_img
        B,C,H,W = img.shape
        out_dim = H*W*C//self.num_patches
        B,N,C = x.shape
        print(f"Shape of x: {x.shape}")
        vab = ViTAttention(embed_dim=self.embed_dim,out_dim=out_dim,num_heads=self.num_heads).to(self.device)
        y, attn = vab(x)
        test_backprop(vab, x)
        print(f"Shape of y: {y.shape}")
        print(f"Shape of attn: {attn.shape}")
        assert y.shape == (B,N,out_dim), f"Expected shape: {(B,N,out_dim)} but got {y.shape}"
        assert attn.shape == (self.batch_size,self.num_heads,self.num_patches,self.num_patches), f"Expected shape: {(self.batch_size,self.num_patches,out_dim)} but got {attn.shape}"
        return x,vab
    @time_func
    def test_ViTPatchEmbeddings(self):
        x = self.ex_img
        vpe = ViTPatchEmbeddings(img_size=self.width,
                                patch_size=self.patch_size,
                                embed_dim=self.embed_dim,
                                in_chans=self.in_channels).to(self.device)
        y = vpe(x)
        test_backprop(vpe, x)
        print(f"Shape of y: {y.shape}")
        assert y.shape == (self.batch_size,self.num_patches,self.embed_dim), f"Expected shape: {(self.batch_size,self.num_patches,self.embed_dim)} but got {y.shape}"
        return x,vpe
    @time_func
    def test_ViTPositionalEmbedding(self):
        x = self.patched_img
        vpose = ViTPositionalEmbedding(embed_dim=self.embed_dim,
                                        num_patches=self.num_patches).to(self.device)
        y = vpose(x)
        test_backprop(vpose, x)
        print(f"Shape of y: {y.shape}")
        assert y.shape == (self.batch_size,self.num_patches,self.embed_dim), f"Expected shape: {(self.batch_size,self.num_patches,self.embed_dim)} but got {y.shape}"
        return x,vpose
    @time_func
    def test_ViTAttentionBlock(self):
        x = self.ex_img
        vab = ViTAttentionBlock(img_size=self.width,
                                in_chans=self.in_channels,
                                embed_dim=self.embed_dim,
                                num_heads=self.num_heads,
                                patch_size=self.patch_size,).to(self.device)
        y,attn = vab(x)
        test_backprop(vab, x)
        pdf_path = visualize_model(vab, x)
        print(f"Shape of y: {y.shape}")
        print(f"Shape of attn: {attn.shape}")
        # Should be same size:
        assert(y.shape == x.shape), f"Expected shape: {x.shape} but got {y.shape}"
        return x,vab
    @time_func
    def test_UNetBlock(self):
        x = self.ex_img
        B,C,H,W = x.shape
        unet = UNetBlock(depth=self.encoder_block_depth,
                        in_channels=self.in_channels,
                        out_channels=self.out_channels,
                        kernel_size=3,
                        stride=1,).to(self.device)
        y,_ = unet(x)
        test_backprop(unet, x)
        print(f"Shape of y: {y.shape}")
        assert(y.shape == (B,self.out_channels,H//2,W//2)), f"Expected shape: {(B,self.out_channels,H//2,W//2)} but got {y.shape}"
        return x,unet

    @time_func
    def test_UNetBlockWithAttention(self):
        x = self.ex_img
        B,C,H,W = x.shape

        unet = UNetBlock(depth=self.encoder_block_depth,
                        in_channels=self.in_channels,
                        out_channels=self.out_channels,
                        kernel_size=3,
                        stride=1,
                        attention=True,
                        img_size=(H,W),
                        patch_size=self.patch_size,
                        num_heads=self.num_heads,
                        embed_dim=self.embed_dim).to(self.device)
        y,_ = unet(x)
        test_backprop(unet, x)
        visualize_model(unet, x)
        print(f"Shape of y: {y.shape}")
        assert(y.shape == (B,self.out_channels,H//2,W//2)), f"Expected shape: {(B,self.out_channels,H//2,W//2)} but got {y.shape}"
        return x,unet
    @time_func
    def test_UNetEncoder(self):
        config_file = "../models/model_configs/UNet.yaml"
        unet_enc = UNetEncoder(config=config_file).to(self.device)
        x = self.ex_img
        print("Input shape: ", x.shape)
        y = unet_enc(x)
        test_backprop(unet_enc, x)
        visualize_model(unet_enc, x)
        return x,unet_enc
    # @time_func
    # def test_CrossAttention(self):
    #     x = self.patched_img
    #     B,N,C = x.shape
    #     C_context = C//2
    #     H_context = H*2
    #     W_context = W*2
    #     context = T.rand(B,C_context,H_context,W_context).to(self.device)
    #     out_dim = H*W*C_context//self.num_patches
    #     ca = CrossAttention(embed_dim=self.embed_dim,
    #                         num_heads=self.num_heads,
    #                         out_dim=out_dim).to(self.device)
    #     y,attn = ca(x,context)
    #     # test_backprop(ca, x)
    #     # visualize_model(ca, x)
    #     print(f"Shape of y: {y.shape}")
    #     print(f"Shape of attn: {attn.shape}")
    #     assert(y.shape == (B,C_context,H,W)), f"Expected shape: {(B,C_context,H,W)} but got {y.shape}"
    #     return x,ca
    @time_func
    def test_CrossAttentionBlock(self):
        x = self.ex_img
        B,C,H,W = x.shape
        C_context = C//2
        H_context = H*2
        W_context = W*2
        context = T.rand(B,C_context,H_context,W_context).to(self.device)
        cab = CrossAttentionBlock(img_size=(H,W),
                                context_size=(H_context,W_context),
                                in_chans=C,
                                context_chans=C_context,
                                embed_dim=self.embed_dim,
                                num_heads=self.num_heads,
                                patch_size=self.patch_size).to(self.device)
        y,attn = cab(x,context)
        correct_input = (x,context)
        # test_backprop(cab, (x,context))
        visualize_model(cab, (x,context))
        print(f"Shape of y: {y.shape}")
        print(f"Shape of attn: {attn.shape}")
        assert(y.shape == (B,C_context,H_context,W_context)), f"Expected shape: {(B,C_context,H_context,W_context)} but got {y.shape}"
        return correct_input,cab
    @time_func
    def test_DecoderBlock(self):
        x = self.ex_img
        B,C,H,W = x.shape
        context = T.randn(B,C//2,H*2,W*2).to(self.device)
        db = DecoderBlock(img_size=(H,W),
                        in_channels=C,
                        out_channels=C//2,
                        context_size=(H*2,W*2),
                        context_channels=C//2,
                        attention=True,
                        embed_dim=self.embed_dim,
                        num_heads=self.num_heads,
                        patch_size=self.patch_size,).to(self.device)
        y,attn = db(x,context)
        correct_input = (x,context)
        assert y.shape == (B,C//2,H*2,W*2), f"Expected shape: {(B,C//2,H*2,W*2)} but got {y.shape}"
        return correct_input,db
    @time_func
    def test_UNetDecoder(self):
        config_file = "../models/model_configs/UNet.yaml"
        unet_enc = UNetEncoder(config=config_file, in_channels=self.in_channels).to(self.device)
        x = self.ex_img
        x_s = unet_enc(x)
        for i in range(len(x_s)):
            print(f"Shape of x_{i}: {x_s[i].shape}")
        print("Input shape: ", x.shape)
        print("Output shape: ", unet_enc.latent_size)
        unet_dec = UNetDecoder(config=config_file,latent_size=unet_enc.latent_size).to(self.device)
        y,y_s,attn = unet_dec(copy(x_s))
        print(len(x_s)," ",len(y_s))
        visualize_model(unet_dec, copy(x_s))
        return x_s,unet_dec
    
    @time_func
    def test_UNet(self):
        config_file = "../models/model_configs/UNet.yaml"
        unet = UNet(config=config_file, in_channels=self.in_channels,verbose=True).to(self.device)
        x = self.ex_img[:1,...]
        y,y_s,attn_s = unet(x)
        print(f"Shape of y: {y.shape}")
        # print(f"Shape of last attn: {attn_s[-1].shape}")
        visualize_model(unet, x)
        return x,unet
class TestGUI:
    def __init__(self):
        testers = Tester(PARAMS)
        self.methods = testers.methods
        self.completion_history = []
        self.logger = TestLogger(SCRIPT_NAME)
        self.layout = [[sg.Text("Select input parameters")],
                       [self.sliders()],
                       [sg.Button("Reset")],
                       [sg.Text("Select methods to test")],
                       [sg.Listbox(values=self.methods, size=(40, 20), key="-LIST-"),sg.Checkbox("All",key="-ALL-",enable_events=True)],
                       [sg.Button("Run"), sg.Button("Exit")]]
        self.window = sg.Window("Testing", self.layout)
        self.run()
    def run(self):
        while True:
            event, self.values = self.window.read()
            if event == sg.WIN_CLOSED or event == "Exit":
                break
            if event == "-ALL-":
                # Select all methods in list by updating background color
                self.window["-LIST-"].BackgroundColor = "green" if self.values["-ALL-"] else None
                self.window["-LIST-"].update(values=self.methods)
                self.window.refresh()
            if event == "Run":
                if self.values["-ALL-"]:
                    self.test_methods(self.methods,True)
                elif self.values["-LIST-"]:
                    self.test_methods(self.values["-LIST-"],False)
            if event == "Reset":
                self.reset_sliders()

        self.display_results()
        self.window.close()
    def test_methods(self,methods,all_methods:bool):
        self.tester = Tester(self.values)
        for method in methods:
            self.logger.initialize_test(method)
            try:
                inp,model = getattr(self.tester,"test_"+method)()
                test_backprop(model, copy(inp))
                flop_counter(model, copy(inp))
                memory_usage(model)
                del model,inp
                empty_cache()
                self.logger.completed_test(method)
                self.completion_history.append((method,True,None))
            except Exception as e:
                self.logger.failed_test(method)
                self.completion_history.append((method,False,e))
        if all_methods:
            self.display_results()
    def display_results(self):
        if len(self.completion_history) == 0:
            self.logger("Results:")
        for method,success,excep in self.completion_history:
            if success:
                msg = f"{method} -> {Fore.GREEN}{'Completed'}{Style.RESET_ALL}"
                self.logger(msg)
            else:
                msg = f"{method} -> {Fore.RED}{'Failed'}{Style.RESET_ALL}: {excep}"
                self.logger.warning(msg)
        self.completion_history = []
    def sliders(self):
        sliders = []
        fixed_length = 15
        for param,val in PARAMS.items():
            if param != "width" and param != "height":
                sliders.append([sg.Text(param),sg.Slider(range=(1,val*5),resolution=1,default_value=int(val),key=str(param),orientation="horizontal",enable_events=True)])
                # sliders.append([sg.Slider(range=(0,val*5),resolution=1,default_value=int(val),key=str(param),orientation="horizontal",enable_events=True)])
            else:
                # print("Skipping")
                sliders.append([sg.Text(param,pad=(fixed_length-len(param))),sg.Slider(range=(0,val*5),resolution=PARAMS["patch_size"],default_value=int(val),key=str(param),orientation="horizontal",enable_events=True)])
                # sliders.append([sg.Slider(range=(0,val*5),resolution=PARAMS["patch_size"],default_value=int(val),key=str(param),orientation="horizontal",enable_events=True)])
        return sg.Column(sliders,vertical_alignment="top",justification="left")
    def reset_sliders(self):
        for param,val in PARAMS.items():
            self.window[param].update(val)
    def reset(self):
        # self.logger.reset()
        self.completion_history = []
    # def closest_multiple(self, n, m):
    #     return int(m * round(float(n)/m))
if __name__=="__main__":
    tester = TestGUI()
