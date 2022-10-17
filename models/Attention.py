import sys
import os
if __name__=="__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.visualizations import visualize_attention_map
from utils.helper import to_2tuple
import math
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

class ViTPositionalEmbedding(nn.Module):
    """
    Create a patch embedding from a patch size.
    """
    def __init__(self,embed_dim,num_patches):
        super().__init__()
        self.pos_embed = nn.Parameter(T.zeros(1, num_patches, embed_dim))
    def forward(self,x):
        x = x + self.pos_embed
        return x

class ViTPatchEmbeddings(nn.Module):
    """
    Vision transformer patch embedding:
    """
    def __init__(self, 
                img_size=256, 
                patch_size=16, 
                in_chans=3, 
                embed_dim=768):
        super().__init__()
        assert img_size is not None, "Image size must be specified when using ViT patch embedding"
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        # print(f"img_size: {img_size}, patch_size: {patch_size}, num_patches: {num_patches}")
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = ViTPositionalEmbedding(embed_dim, num_patches)
    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.pad_to_patch(x)
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.pos_embed(x)
        return x
    def pad_to_patch(self, x):
        B, C, H, W = x.shape
        pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
        pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
        x = F.pad(x, (0, pad_w, 0, pad_h))
        return x

class ViTAttention(nn.Module):
    """
    Vision transformer attention block:
    """
    def __init__(self, 
                embed_dim,
                out_dim=256*256*3,
                num_heads=8, 
                qkv_bias=False, 
                qk_scale=None, 
                attn_drop=0., 
                proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, **kwargs):
        B, N, C = x.shape
        assert C%self.num_heads == 0, f"Embedding dimension {C} is not divisible by number of heads {self.num_heads}"
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # print(f"Self attention qkv shape: {qkv.shape}")
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # print(f"Self attention output shape: {x.shape}")
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn
class ViTAttentionBlock(nn.Module):
    """
    Block in the pipeline which attends to all the patches in the image.
    """
    def __init__(self,
                img_size=None,
                in_chans=3,
                out_chans=3,
                embed_dim=768,
                num_heads=8,
                patch_size=16,
                qkv_bias=False,
                qk_scale=None,
                attn_drop=0.,
                proj_drop=0.):
        super().__init__()
        # embed_dim = img_size*img_size*in_chans
        assert img_size is not None, "img_size must be specified"
        img_size = to_2tuple(img_size)
        # print(f"Attention block: img_size: {img_size}, patch_size: {patch_size}, in_chans: {in_chans}, embed_dim: {embed_dim}")
        self.patch_embedder = ViTPatchEmbeddings(img_size=img_size,embed_dim=embed_dim,patch_size=patch_size,in_chans=in_chans)
        self.num_patches = self.patch_embedder.num_patches
        out_dim = img_size[0]*img_size[1]*out_chans//self.num_patches
        self.attention = ViTAttention(embed_dim,out_dim,num_heads,qkv_bias,qk_scale,attn_drop,proj_drop)
    def forward(self,img):
        B, C, H, W = img.shape
        x = self.patch_embedder(img)
        x, attn = self.attention(x) # x shape: (bs, num_patches, embed_dim), attn shape: (bs, num_heads, num_patches, num_patches)
        
        x = x.reshape(B,C,H,W)
        return x, attn


class UpSampleBlock(nn.Module):
    """
    Block in the pipeline which upsamples the image.
    """
    def __init__(self,
                in_chans:int,
                out_chans:int,
                activation:Union[tuple,int]=nn.Sigmoid(),
                scale=2):
        super().__init__()
        self.conv = nn.Conv2d(in_chans,out_chans,1,padding=0)
        self.activation = activation
        self.upsample = nn.Upsample(scale_factor=scale,mode='bilinear',align_corners=True)
    def forward(self,x):
        x = self.conv(x)
        x = self.upsample(x)
        if self.activation is not None:
            x = self.activation(x)
        return x 


class CrossAttention(nn.Module):
    def __init__(self, 
                embed_dim,
                out_dim=256*256*3,
                num_heads=8, 
                qkv_bias=False, 
                qk_scale=None, 
                attn_drop=0., 
                proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context, **kwargs):
        B, N, C = x.shape
        assert C%self.num_heads == 0, f"Embedding dimension {C} is not divisible by number of heads {self.num_heads}"
        x_context = T.cat([x,context],dim=1)
        qkv = self.qkv(x_context)
        # print(f"qkv shape: {qkv.shape}")
        qkv = qkv.reshape(B, N, 3, self.num_heads, 2*C // self.num_heads).permute(2, 0, 3, 1, 4)
        # print(f"Cross attention qkv shape: {qkv.shape}")
        q, k, v = qkv[0], qkv[1], qkv[2]
        # print(f"q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # print(f"attn shape: {attn.shape}")
        x = (attn @ v).transpose(1, 2).reshape(B, 2*N, C)
        # print(f"x shape: {x.shape}")
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn
class CrossAttentionBlock(nn.Module):
    """
    Block in the pipeline which attends to all the patches in the image.
    """
    def __init__(self,
                img_size:tuple=None,
                context_size:tuple=None,
                in_chans:int=3,
                context_chans:int=3,
                embed_dim:int=768,
                num_heads:int=8,
                patch_size:int=16,
                qkv_bias:bool=False,
                qk_scale:bool=None,
                attn_drop:bool=0.,
                proj_drop:bool=0.,
                scale:int=2):
        super().__init__()
        # embed_dim = img_size*img_size*in_chans
        assert img_size is not None, "img_size must be specified"
        assert context_size is not None, "context_size must be specified"
        img_size = to_2tuple(img_size)
        # print(f"Attention block: img_size: {img_size}, patch_size: {patch_size}, in_chans: {in_chans}, embed_dim: {embed_dim}")
        self.scale = scale
        self.patch_embedder = ViTPatchEmbeddings(img_size=img_size,embed_dim=embed_dim,patch_size=patch_size,in_chans=in_chans)
        self.context_embedder = nn.Sequential(nn.MaxPool2d(2,2),ViTPatchEmbeddings(img_size=img_size,embed_dim=embed_dim,patch_size=patch_size,in_chans=context_chans))
        self.num_patches = self.patch_embedder.num_patches
        out_dim = img_size[0]*img_size[1]*context_chans//(self.num_patches*2)
        self.up_sample = UpSampleBlock(in_chans=context_chans,out_chans=context_chans,scale=scale)
        self.attention = CrossAttention(embed_dim,out_dim,num_heads,qkv_bias,qk_scale,attn_drop,proj_drop)
    def forward(self,img,context):
        B, C, H, W = img.shape
        _, C_context, H_context, W_context = context.shape
        x = self.patch_embedder(img)
        context_emb = self.context_embedder(context)
        # print(f"x shape: {x.shape}, context_emb shape: {context_emb.shape}")

        x, attn = self.attention(x,context_emb) # x shape: (bs, num_patches, embed_dim), attn shape: (bs, num_heads, num_patches, num_patches)
        x = x.reshape(B,C_context,H,W)
        x = self.up_sample(x) # x shape: (bs, C_context, H*scale, W*scale)
        x = T.mul(x,context) # element-wise multiplication => x shape: (bs, C_context, H_context, W_context)
        return x, attn