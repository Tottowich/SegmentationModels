# SegmentationModels
This repository contains code for training and inference of semantic segmentation models in Pytorch.
The main idea behind this repository is to make it easy to modify models from [U-Net Transformer: Self and Cross Attention for Medical Image Segmentation](https://arxiv.org/abs/2103.06104) by Oliver Petit, Nicloas THo,e. Clement Rambour and Luc Soler.
The idea was to make the models configurable using a GUI or a config file. This way it is easy to try out different models and configurations without having to write a lot of code. The GUI is made using [PySimpleGUI](https://pysimplegui.readthedocs.io/en/latest/). The config file is made using [PyYAML](https://pyyaml.org/wiki/PyYAMLDocumentation). 

## Model Architecture

The model architecture is based on the U-Net Transformer architecture from the paper [U-Net Transformer: Self and Cross Attention for Medical Image Segmentation](https://arxiv.org/abs/2103.06104). The model is made up of a U-Net architecture with self attention blocks and cross attention blocks. The self attention blocks are both within the decoder and within the encoder. The cross attention blocks are between the encoder and decoder. The model is made up of the following blocks:

- Encoder
    - Encoder Block - Modifiable depth of the encoder
        - Convolution Block - Modifiable depth per block
            - Convolution
            - Batch Normalization
            - 
        - Multi Head Self Attention 
