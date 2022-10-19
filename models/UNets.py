from json import encoder
import sys
import os
from typing import NamedTuple
from typing import Union, Tuple, Callable, Optional, List
# if __name__=="__main__":
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from .Attention import ViTAttentionBlock
from .Attention import CrossAttentionBlock
from utils.visualizations import visualize_attention_map
from utils.helper import to_2tuple
from utils.helper import ModelLogger
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import yaml
"""
ViTAttentionBlock:
Takes as input a tensor of shape (B, C, H, W) and returns a tensor of shape (B, C, H, W).
"""
class StrToActivation(nn.Module):
    def __init__(self,activation:str):
        super().__init__()
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(inplace=True)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "softmax":
            self.activation = nn.Softmax(dim=1)
        elif activation == "identity":
            self.activation = nn.Identity()
        else:
            raise Exception(f"Unknown activation: {activation}, available: relu, leaky_relu, tanh, sigmoid, softmax, identity")
    def forward(self,x):
        return self.activation(x)
# StrToActivationDict = {
#     "relu": nn.ReLU(inplace=True),
#     "leaky_relu": nn.LeakyReLU(inplace=True),
#     "tanh": nn.Tanh(),
#     "sigmoid": nn.Sigmoid(),
#     "softmax": nn.Softmax(dim=1),
#     "identity": nn.Identity()
# }

class TransposeUpsample(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:Tuple[tuple,int]=1, stride:Tuple[tuple,int]=1, padding:Tuple[tuple,int]=1, output_padding:Tuple[tuple,int]=0):
        """
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (tuple or int): Size of the convolving kernel
            stride (tuple or int): Stride of the convolution. Default: 1
            padding (tuple or int): Zero-padding added to both sides of the input. Default: 0
            output_padding (tuple or int): A zero-padding of size (out_pad_h, out_pad_w) will be added to both sides of the output. Default: 0
        """
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class Conv1x1(nn.Module):
    def __init__(self, in_channels:int, out_channels:int,reps:int=1,activation:nn.Module=None,dropout:float=0.0):
        """
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (tuple or int): Size of the convolving kernel
            stride (tuple or int): Stride of the convolution. Default: 1
            padding (tuple or int): Zero-padding added to both sides of the input. Default: 0
            output_padding (tuple or int): A zero-padding of size (out_pad_h, out_pad_w) will be added to both sides of the output. Default: 0
        """
        super().__init__()
        blocks = []
        if activation is None:
            activation = nn.Identity()
        for _ in range(reps):
            blocks.append(nn.Conv2d(in_channels, out_channels, 1,1,0))
            blocks.append(nn.BatchNorm2d(out_channels))
            blocks.append(activation)
            blocks.append(nn.Dropout(dropout))
        self.conv = nn.Sequential(*blocks)
        # self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        # self.bn = nn.BatchNorm2d(out_channels)
        # self.dropout = nn.Dropout2d(dropout)
        # self.activation = activation if not None else nn.Identity()
    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        # x = self.dropout(x)
        # x = self.activation(x)
        return x
class Conv3x3(nn.Module):
    def __init__(self, in_channels:int, out_channels:int,reps:int=1,activation:nn.Module=None,dropout:float=0.0,norm:bool=True):
        """
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution.
            reps: (int) Number of times to repeat the convolution.
            activation (nn.Module): Activation function to be used after the convolution.
            dropout (float): Dropout rate to be used after the convolution.
        """
        super().__init__()
        blocks = []

        for _ in range(reps):
            blocks.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1))
            blocks.append(nn.BatchNorm2d(out_channels) if norm else nn.Identity())
            blocks.append(activation if activation is not None else nn.Identity())
            blocks.append(nn.Dropout2d(dropout))
        self.conv = nn.Sequential(*blocks)
        # self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1) # Same padding
        # self.bn = nn.BatchNorm2d(out_channels)
        # self.dropout = nn.Dropout2d(dropout)
        # self.activation = activation if not None else nn.Identity()
    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        # x = self.dropout(x)
        # x = self.activation(x)
        return x
class EncoderBlock(nn.Module):
    depth = 0
    def __init__(self, 
                in_channels:int, 
                out_channels:int, 
                kernel_size:int, 
                stride:int, 
                padding:Union[str,int],
                norm:bool=True,
                act:Union[bool,nn.Module]=True,
                pool:Union[bool,nn.Module]=False,
                dropout:float=0.1,):
        """
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int): Size of the convolving kernel
            stride (int): Stride of the convolution. Default: 1
            padding (str or int): Padding added to both sides of the input before convolving. Default: 0
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels) if norm else nn.Identity()
        self.act = self._activation(act)
        self.max_pool = self._pooling(pool)
        self.dropout = nn.Dropout(dropout)
        self.seq_depth = self.depth
    def forward(self, x):
        # print(f"EncoderBlock x: {x.shape} @ {self.seq_depth}")
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.max_pool(x)
        return x
    def _pooling(self,pool):
        # print(f"pool: {pool}")
        if pool:
            if isinstance(pool, bool):
                # print(f"pool bool: {pool}")
                return nn.MaxPool2d(2, 2)
            elif isinstance(pool, int):
                # print(f"pool int: {pool}")
                return nn.MaxPool2d(pool, pool)
            elif isinstance(pool, nn.Module):
                # print(f"pool Module: {pool}")
                return pool
        else:
            return nn.Identity()
    def _activation(self,act):
        if isinstance(act, bool) and act:
            return nn.ReLU(inplace=True)
        elif isinstance(act, nn.Module):
            return act
        else:
            return nn.Identity()

class UNetBlock(nn.Module):
    def __init__(self, 
                depth:int, 
                in_channels:int, 
                out_channels:int, 
                kernel_size:Union[int,list[int]]=3, 
                stride:Union[int,list[int]]=1, 
                padding:Union[str,int]=1,
                norms:list[Union[bool,nn.Module]]=None,
                acts:list[Union[bool,nn.Module]]=None,
                dropout:float=0.1,
                pool:list[Union[bool,nn.Module]]=None,
                attention:bool=False,
                img_size:Union[int,tuple]=None,
                embed_dim:int=768,
                num_heads:int=8,
                patch_size:int=16,
                attn_drop:float=0.1,
                ):
        super().__init__()
        # Encoder Block Specific
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pool = pool if isinstance(pool,list) else [False]*(depth-1) + [True]
        self.norms = norms if isinstance(norms,list) else [True]*(depth)
        self.acts = acts if isinstance(acts,list) else [True]*(depth)
        self.dropout = dropout
        # Attention Specific
        self.attention = attention
        self.img_size = to_2tuple(img_size)
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.attn_drop = attn_drop
        self.embed_dim = embed_dim
        self.encoder_block = self.build_block()
        self.attn_block = self.build_attention()
    def build_attention(self)->Union[ViTAttentionBlock,nn.Identity]:
        if self.attention:
            assert self.img_size is not None, "Image size must be specified for attention"
            # attn_img_size = (self.img_size[0]//self.patch_size, self.img_size[1]//self.patch_size) if self.pool[-1] else self.img_size
            # print(f"Attention Image Size: {self.img_size}")
            return ViTAttentionBlock(
                img_size=self.img_size,
                in_chans=self.out_channels,
                out_chans=self.out_channels,
                patch_size=self.patch_size,
                num_heads=self.num_heads,
                attn_drop=self.attn_drop,
                embed_dim=self.embed_dim,
            )
        else:
            def identity_wrapper(x):
                return x,None
            return identity_wrapper
    def build_block(self):
        block = []
        EncoderBlock.depth = 0
        in_channels = self.in_channels
        for i,(norm,act,pool) in enumerate(zip(self.norms,self.acts,self.pool)):
            # print(f"In channels -> {in_channels} -> Out channels: {self.out_channels}")
            block.append(EncoderBlock(
                in_channels=in_channels,# if not i else self.out_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                norm=norm,
                act=act,
                pool=pool,
                dropout=self.dropout,
            ))
            EncoderBlock.depth += 1
            in_channels = self.out_channels
        return nn.Sequential(*block)
    def forward(self, x):
        # print(f"UNetBlock: {x.shape}")
        x = self.encoder_block(x)
        # print(f"EncoderBlock out: {x.shape}")
        x, attn = self.attn_block(x)
        # print(f"EncoderBlock out attn: {x.shape}")
        # print(f"AttentionBlock out: {x.shape}")
        return x,attn
class UNetEncoder(nn.Module):
    """
    A UNet encoder with possibility of using vision attention.
    The structure of the UNet encoder should be read from a yaml file.
    The .yaml file should contain.
    A dictionary the dictionary should contain the following keys:
        encoder_depth: int - the depth of the encoder
        depth: list[int] containing how many EncoderBlocks should be in each UNetBlock.
        kernel size: list[int or tuple] containing what kernel size of each EncoderBlock.
        out_channels: list[int] containing what out_channels of each EncoderBlock.
        stride: list[int or tuple] containing what stride of each EncoderBlock.
        padding: list[int or tuple] containing what padding of each EncoderBlock.
        norms: list[bool or nn.Module] containing what normalization of each EncoderBlock.
        acts: list[bool or nn.Module] containing what activation of each EncoderBlock.
        pool: list[bool or nn.Module] containing what pooling of each EncoderBlock.
        dropout: list[float] containing what dropout of each EncoderBlock.
        attention: list[bool] containing if attention should be used in each UNetBlock.
        img_size: int initial img size
        embed_dim: list[int] containing the embedding dimension of each UNetBlock.
        num_heads: list[int] containing the number of heads of each UNetBlock.
        patch_size: list[int] containing the patch size of each UNetBlock.
        attn_drop: list[float] containing the dropout of each UNetBlock.
    """
    def __init__(self, config:Union[dict,str],verbose:bool=False):
        super().__init__()
        if isinstance(config,dict):
            self.in_channels = config.get("in_channels",3)
            self._config = config["encoder"]

        else:
            self._config = self.read_config(config)
        self.depth = self._config["depth"]
        self.kernel_size = self._config["kernel_size"]
        self.out_channels = self._config["out_channels"]
        self.stride = self._config["stride"]
        self.padding = self._config["padding"]
        self.norms = self._config["norms"]
        self.acts = self._config["acts"]
        self.pool = self._config["pool"]
        self.dropout = self._config["dropout"]
        self.attention = self._config["attention"]
        self.img_size = to_2tuple(self._config["img_size"])[:2]
        self.embed_dim = self._config["embed_dim"]
        self.num_heads = self._config["num_heads"]
        self.patch_size = self._config["patch_size"]
        self.attn_drop = self._config["attn_drop"]
        self.outputs = []
        self.verbose = verbose
        if self.verbose:
            self.logger = ModelLogger()
        self.encoder = self.build_encoder()
    def build_encoder(self):
        encoder = []
        output_img_size = self.img_size
        in_channels = self.in_channels
        self.attention_layers = []
        for i,(depth,
               kernel_size,
               out_channels,
               stride,
               padding,
               norm,
               act,
               pool,
               dropout,
               attention,
               embed_dim,
               num_heads,
               patch_size,
               attn_drop) in enumerate(zip(
                                self.depth,
                                self.kernel_size,
                                self.out_channels,
                                self.stride,
                                self.padding,
                                self.norms,
                                self.acts,
                                self.pool,
                                self.dropout,
                                self.attention,
                                self.embed_dim,
                                self.num_heads,
                                self.patch_size,
                                self.attn_drop)):
            output_img_size = (output_img_size[0]//2,output_img_size[1]//2) if pool else output_img_size
            encoder.append(UNetBlock(
                depth=depth, 
                in_channels=in_channels, 
                out_channels=out_channels,
                kernel_size = kernel_size, 
                stride=stride, 
                padding=padding,
                norms=norm,
                acts=act,
                dropout=dropout,
                pool=pool,
                attention=attention,
                img_size=output_img_size,
                embed_dim=embed_dim,
                num_heads=num_heads,
                patch_size=patch_size,
                attn_drop=attn_drop,))
            in_channels = out_channels
            if attention:
                self.attention_layers.append(i)
            if self.verbose:
                self.logger.info(f"UNetBlock {i} out size: {output_img_size}")
        self.latent_size = output_img_size
        return nn.ModuleList(encoder) #nn.Sequential(*encoder)
    def read_config(self,config_file:str):
        with open(config_file) as f:
            config = yaml.load(f,Loader=yaml.FullLoader)
        self._steps = config["steps"]
        self.in_channels = config["in_channels"]
        encoder_config = config["encoder"]
        for key,item in encoder_config.items():
            # if key in ["depth","kernel_size","out_channels","stride","padding","norms","acts","pool","dropout","attention","embed_dim","num_heads","patch_size","attn_drop"]:
            if not isinstance(item,list):
                encoder_config[key] = [item]*self._steps
            else:
                if len(item) != self._steps:
                    # print(f"Number of \'{key}\' provided does not match encoder depth. Setting to default.")
                    # Extent by last value if not enough
                    if len(item) < self._steps:
                        encoder_config[key] = item + [item[-1]]*(self._steps-len(item))
                    else:
                        encoder_config[key] = item[:self._steps]
        encoder_config["in_channels"] = self.in_channels
        return encoder_config
    def forward(self, x):
        # x = self.encoder(x)
        # print("Num blocks: ",len(self.encoder))x
        x_s = [x]
        for i,block in enumerate(self.encoder):
            if self.verbose:
                self.logger.debug(f"EncoderBlock {i+1}; x: {x.shape}")
            x, attn = block(x)
            # self.outputs.append((x,attn))
            x_s.append(x)
        return x_s
    @property
    def test_input(self):
        return T.rand(1, self.in_channels, *self.img_size[:2])
    @property
    def device(self):
        return next(self.parameters()).device
    @property
    def encoder_config(self)->dict:
        return self._config
    @property
    def encoder_depth(self)->int:
        return self.depth
    @property
    def steps(self)->int:
        if hasattr(self,"_steps"):
            return self._steps
        else:
            return len(self.encoder)
class UpSampleBlock(nn.Module):
    def __init__(self,
                in_channels:int,
                out_channels:int,
                scale_factor:float=2,
                ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels 
        self.scale_factor = scale_factor
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.conv3x3 = Conv3x3(in_channels,out_channels)
    def forward(self,x):
        x = self.upsample(x)
        x = self.conv3x3(x)
        return x
class DecoderBlock(nn.Module):
    block = 0
    def __init__(self,
                in_channels:int,
                out_channels:int,
                img_size:tuple,
                context_channels:int,
                context_size:tuple=None,
                attention:bool=True,
                embed_dim:int=768,
                num_heads:int=8,
                patch_size:int=16,
                attn_drop:float=0.0,
                conv1x1:bool=1,
                conv3x3:int=1,
                activation:nn.Module=None,
                ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_size = img_size
        self.context_size = context_size if context_size else 2*img_size
        self.context_channels = context_channels
        self.attention = attention
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.attn_drop = attn_drop
        self.upsample = UpSampleBlock(in_channels,in_channels)
        # self.attn_proj = Conv1x1()
        # print(f"Activation Decoder: {activation}")
        self.conv_attn = Conv1x1(in_channels,in_channels,reps=conv1x1,activation=activation) if attention else nn.Identity()
        # self.conv1x1 = Conv1x1(in_channels,out_channels,reps=conv1x1) if conv1x1 else nn.Identity()
        self.conv1x1 = Conv1x1(in_channels,context_channels,reps=conv1x1,activation=activation) if conv1x1 else nn.Identity()
        # self.conv3x3 = Conv3x3(in_channels,out_channels,reps=conv3x3) if conv3x3 else nn.Identity()
        self.conv3x3 = Conv3x3(2*context_channels,out_channels,reps=conv3x3,activation=activation) if conv3x3 else nn.Identity()
        # self.norm = nn.LayerNorm(out_channels) Maybe??
        if attention:
            self.mhca = self._build_attention() if attention else nn.Identity()
        else:
            self.skip = Conv1x1(context_channels,context_channels)
        self.attention = attention
    def _build_attention(self):
        assert self.attention, "Attention not enabled"
        assert self.context_size is not None, "Context size not specified"
        assert self.context_channels is not None, "Context channels not specified"
        return CrossAttentionBlock(
            img_size=self.img_size,
            context_size=self.context_size,
            in_chans=self.in_channels,
            context_chans=self.context_channels,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            patch_size=self.patch_size,
            attn_drop=self.attn_drop,
            )
    def forward(self,x_in,context):
        # print(f"\nDecoderBlock {DecoderBlock.block}: {x_in.shape} {context.shape}")
        # DecoderBlock.block += 1
        # print(f"DecoderBlock x_in: {x_in.shape}")
        # print(f"DecoderBlock context: {context.shape}")
        # # if self.attention:
        x = self.conv_attn(x_in)
        # print(f"DecoderBlock x post conv_attn: {x.shape}")
        if self.attention:
            x_attn,attn = self.mhca(x,context)
            # print(f"DecoderBlock x_attn: {x_attn.shape}")
            # print(f"DecoderBlock attn: {attn.shape}")
        else:
            x_attn = self.skip(context) # Regular skip connection
            attn = None
        x = self.upsample(x_in)
        # print(f"DecoderBlock x post upsample: {x.shape}s")

        x = self.conv1x1(x)
        # print(f"DecoderBlock x post 1x1: {x.shape}")
        # print(f"DecoderBlock x_attn: {x_attn.shape}")
        x = T.cat([x,x_attn],dim=1)
        # print(f"DecoderBlock x post cat: {x.shape}")
        x = self.conv3x3(x)
        # print(f"DecoderBlock x post 3x3: {x.shape}")
        return x, attn
class OutputBlock(nn.Module):
    def __init__(self,
                in_channels:int,
                out_channels:int,
                conv1x1:bool=1,
                conv3x3:int=1,
                ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1x1 = Conv1x1(in_channels,out_channels,reps=conv1x1,activation=nn.Softmax(dim=1),norm=False) if conv1x1 else nn.Identity()
        self.conv3x3 = Conv3x3(in_channels,out_channels,reps=conv3x3,activation=nn.Softmax(dim=1),norm=False) if conv3x3 else nn.Identity()
    def forward(self,x):
        x = self.conv1x1(x)
        y = self.conv3x3(x)
        return y
class UNetDecoder(nn.Module):
    def __init__(self,
                config:str,
                latent_size:list=None,
                verbose:bool=False,
                return_attention:bool=False,):
        super().__init__()
        self.attention_layers = [] #attention_layers
        self.config_file = config
        if isinstance(config,str):
            self._decoder_config = self.read_config(config)
        else:
            assert isinstance(config,dict), "Config must be a dict or a path to a yaml file"
            self.num_classes = config["num_classes"]
            self.steps = config["steps"]
            self._decoder_config = config["decoder"]
        self._return_attention = return_attention
        # self.decoder_depth = self.decoder_config["decoder_depth"]
        self.attention = self._decoder_config["attention"]
        self.conv1x1s = self._decoder_config["conv1x1"]
        self.conv3x3s = self._decoder_config["conv3x3"]
        self.embed_dim = self._decoder_config["embed_dim"]
        self.num_heads = self._decoder_config["num_heads"]
        self.patch_size = self._decoder_config["patch_size"]
        self.attn_drop = self._decoder_config["attn_drop"]
        self.in_channels = self._decoder_config["in_channels"]
        self.out_channels = self._decoder_config["out_channels"]
        self.latent_size = latent_size
        # self.img_size = self.decoder_config["img_size"]
        # self.context_size = self.decoder_config["context_size"]
        self.context_channels = self._decoder_config["context_channels"]
        self.decoder = self.build_decoder()
        self.verbose = verbose
        if self.verbose:
            self.logger = ModelLogger()
    def read_config(self,config_file):
        with open(config_file) as f:
            config = yaml.load(f,Loader=yaml.FullLoader)
        self.steps = config["steps"]
        self.num_classes = config["num_classes"]
        decoder_config = config["decoder"]
        for key,item in decoder_config.items():
            if True: #key in ["depth","kernel_size","out_channels","stride","padding","norms","dropout","attention","embed_dim","num_heads","patch_size","attn_drop"]:
                if not isinstance(item,list):
                    decoder_config[key] = [item]*self.steps
                else:
                    if len(item) != self.steps:
                        # print(f"Length of s\'{key}\' does not match encoder depth. Setting to default.")
                        # Extent by last value
                        if len(item) < self.steps:
                            item = item + [item[-1]]*(self.steps-len(item))
                        else:
                            item = item[:self.steps]
                        decoder_config[key] = item
                # print(f"Decoder config: {key} = {decoder_config[key]}")
        return decoder_config
    def build_decoder(self):
        decoder = []
        img_size = self.latent_size
        for i in range(self.steps):
            decoder.append(DecoderBlock(
                in_channels=self._decoder_config["in_channels"][i],
                out_channels=self._decoder_config["out_channels"][i] if i < self.steps else self.num_classes,
                img_size=img_size,
                context_channels=self._decoder_config["context_channels"][i],# if i < self.steps-1 else self.num_classes,
                attention=self._decoder_config["attention"][i],
                embed_dim=self._decoder_config["embed_dim"][i],
                num_heads=self._decoder_config["num_heads"][i],
                patch_size=self._decoder_config["patch_size"][i],
                attn_drop=self._decoder_config["attn_drop"][i],
                conv1x1=self._decoder_config["conv1x1"][i],
                conv3x3=self._decoder_config["conv3x3"][i],
                activation=StrToActivation(self._decoder_config["activation"][i]),
                # activation=StrToActivationDict[self.decoder_config["activation"][i]]
                ))
            img_size = [img_size[0]*2,img_size[1]*2]
            self.attention_layers.append(i)
        # Replace last layer with softmax if not already
        decoder.append(OutputBlock(self._decoder_config["out_channels"][-1],self.num_classes,conv1x1=0,conv3x3=1))
        return nn.ModuleList(decoder)
    def forward(self,encoder_outputs:list[T.Tensor]):
        x = encoder_outputs.pop()
        x_s = []
        attn_s = []
        for i,(layer,context) in enumerate(zip(self.decoder,reversed(encoder_outputs))):
            encoder_outputs.pop()
            if self.verbose:
                self.logger.debug(f"DecoderBlockInput {i+1}; x: {x.shape}, context: {context.shape}")
            x, attn = layer(x,context)
            if self._return_attention:
                x_s.append(x.detach().cpu().numpy())
                attn_s.append(attn.detach().cpu().numpy() if attn is not None else None)
        y = self.decoder[-1](x)
        assert len(encoder_outputs) == 0, "Not all encoder outputs have been used"
        return (y,attn_s,x_s) if self._return_attention else (y,None,None)  #
    @property
    def decoder_config(self):
        return self._decoder_config
    @property
    def device(self):
        return next(self.parameters()).device


# output_type = NamedTuple("UNetOutput", [("y",T.Tensor),("attn",Optional[T.Tensor]),("y_s",Optional[T.Tensor])])
class UNet(nn.Module):
    def __init__(self,config:Union[str,dict]=None,device:T.device=None,verbose:bool=False,func:Callable=None,return_attention:bool=False,checkpoint:str=None):
        super().__init__()
        self.func = func
        self.return_attention = return_attention
        self.verbose = verbose

        if checkpoint is None:
            self.build_unet(config=config)
        else:
            assert config is None, "Beware that config is ignored when loading from checkpoint. Use config to build new model."
            self.load(checkpoint)
        # Named type for output list of tensors
        # self.output_type = NamedTuple("UNetOutput", [("y",T.Tensor),("attn",Optional[T.Tensor]),("y_s",Optional[T.Tensor])])
        if device is not None:
            self.to(device)
        if verbose:
            self.logger = ModelLogger()
    def build_unet(self,config:Union[str,dict]):
        """
        Build UNet from config file or dict.
        """
        self.encoder:UNetEncoder = UNetEncoder(config,verbose=self.verbose)
        self.decoder:UNetDecoder = UNetDecoder(config,self.encoder.latent_size,verbose=self.verbose,return_attention=self.return_attention)
    def load(self,path:str):
        """Load model from checkpoint."""
        checkpoint = T.load(path)
        if isinstance(checkpoint,dict):
            config = checkpoint['model_config']
            print('Loading model from checkpoint: {}'.format(path))
            self.build_unet(config) # When loading a previous
            self.load_state_dict(checkpoint['model'])
        else:
            raise ValueError("Checkpoint is not a dict. Cannot load model.")
        return checkpoint
    def select_prediction(self,outputs:list[T.Tensor]):
        # For each pixel select the class with the highest probability
        return T.argmax(outputs,dim=1)
    # @T.no_grad()
    def warmup(self)->None:
        """Warmup the model by running a forward pass with random data."""
        self(self.test_input.to(self.encoder.device))

    # def forward(self,x:T.Tensor)->output_type:
    def forward(self,x:T.Tensor)->tuple[T.Tensor,Optional[T.Tensor],Optional[T.Tensor]]:
        """
        Forward pass of the model.

        If return_attention is True, the attention maps are returned as well as the output data at each decoder step.
        """
        encoder_outputs = self.encoder(x)
        if self.func is not None:
            encoder_outputs = self.func(encoder_outputs)
        y,attn_s,y_s = self.decoder(encoder_outputs)
        return  y,attn_s,y_s
    @property
    def config(self)->dict:
        config = {}
        config["num_classes"] = self.num_classes
        config["in_channels"] = self.encoder.in_channels
        config["steps"] = self.encoder.steps
        config["encoder"] = self.encoder.encoder_config
        config["decoder"] = self.decoder.decoder_config
        return config
    @property
    def device(self)->T.device:
        return next(self.parameters()).device
    @T.no_grad()
    def predict(self,x:T.Tensor)->tuple[T.Tensor,list[T.Tensor],list[T.Tensor]]:
        """
        Predict the output of the model.
        """
        x = x.to(self.device)
        y,attn_s,y_s = self.sample(x)
        y = self.select_prediction(y)
        return y,attn_s,y_s
    @T.no_grad()
    def sample(self,x:T.Tensor)->tuple[T.Tensor,list[T.Tensor],list[T.Tensor]]:
        # Use torch mixed precision
        if self.device == "cuda":
            with T.cuda.amp.autocast():
                y,attn_s,y_s = self(x)
        else:
            y,attn_s,y_s = self(x)
        return y,attn_s,y_s
    @T.no_grad()
    def get_latent(self,x:T.Tensor)->T.Tensor:
        return self.encoder(x)[-1]
    @T.no_grad()
    def get_attentions(self,x:T.Tensor)->list[T.Tensor]:
        encoder_outputs = self.encoder(x)
        return self.decoder(encoder_outputs)[1]
    @property
    def latent_size(self)->tuple[int]:
        return self.encoder.latent_size
    @property
    def test_input(self)->T.Tensor:
        if not hasattr(self,"test_inp"):
            self.test_inp = T.randn(1,self.encoder.in_channels,*self.encoder.img_size[:2],requires_grad=True,device=self.encoder.device)
        else:
            self.test_inp = self.test_inp.to(self.device)
        return self.test_inp
    @property
    def num_classes(self)->int:
        return self.decoder.num_classes
    @property
    def img_size(self)->tuple[int]:
        return self.encoder.img_size
    @property
    def num_parameters(self)->int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    @property
    def num_layers(self)->int:
        return len(self.encoder) + len(self.decoder)
        