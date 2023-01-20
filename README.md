# SegmentationModels
This repository contains code for training and inference of semantic segmentation models in Pytorch.
The main idea behind this repository is to make it easy to modify models from [U-Net Transformer: Self and Cross Attention for Medical Image Segmentation](https://arxiv.org/abs/2103.06104) by Oliver Petit, Nicloas THo,e. Clement Rambour and Luc Soler.
The idea was to make the models configurable using a GUI or a config file. This way it is easy to try out different models and configurations without having to write a lot of code. The GUI is made using [PySimpleGUI](https://pysimplegui.readthedocs.io/en/latest/). The config file is made using [PyYAML](https://pyyaml.org/wiki/PyYAMLDocumentation). 

## Model Architecture

The model architecture is based on the U-Net Transformer architecture from the paper [U-Net Transformer: Self and Cross Attention for Medical Image Segmentation](https://arxiv.org/abs/2103.06104). The model is made up of a U-Net architecture with self attention blocks and cross attention blocks. The self attention blocks are both within the decoder and within the encoder. The cross attention blocks are between the encoder and decoder. The model is made up of the following blocks:
* U-Net
    * Encoder
        * Encoder Block - Modifiable depth of encoder, *steps* in .yaml file
            * Convolution Block - Modifiable depth per block, *encoder.depth*.
                * Convolution - Modifiable convolution per block, *encoder.out_channels*, *encoder.kernel_size*, *encoder.stride* and *encoder.padding*.
                * Batch Normalization 2D (Optional) - Use of batch normalization in convolution block, *encoder.norms*.
                * Activation (Optional) - Use of activation in convolution block, default [ReLU](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html), *encoder.acts*.
                * Dropout (Optional) - Use of dropout in convolution block, *encoder.dropout*.
                * Max Pooling 2D (Optional) - Use of max pooling in convolution block, *encoder.pool*.
            * Multi Head Self Attention (Optional per block) - Use of [visual self attention](https://arxiv.org/abs/2010.11929) at the end of encoder block, *attention*.
                * Patch Embedding - Embedding of patches to be used in self attention, *encoder.embed_dim* and *encoder.patch_size*.
                * Attention - Regular self attention, *encoder.attn_heads*, *encoder.embed_dim*.
                * Dropout - Dropout to be used in self attention, *encoder.attn_drop*.
    * Decoder - Same depth as encoder, *steps* in .yaml file
        * Decoder Block 
            * Context Connection Block (If Attention) - Used for dimensionality correction when using cross attention. Depth of skip connection, *decoder.conv1x1*.
                * Convolution - 1x1 convolution to be used in skip connection.
                * Batch Normalization 2D - Batch normalization to be used in skip connection, always used.
                * Activation - Use of activation same as rest of decoder block. Default [ReLU](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html), *decoder.activation*.
            * Skip Connection Block (If NOT Attention) - Use of skip regular U-Net connection when NOT using cross attention. 
                * Convolution - 1x1 convolution to be used in skip connection.
                * Batch Normalization 2D - Batch normalization to be used in skip connection, always used.
                * Activation - Use of activation same as rest of decoder block. Default [ReLU](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html), *decoder.activation*.
            * Conv1x1 Block - Modifiable depth per block, *decoder.conv1x1*.
                * Convolution - 1x1 convolution from input channels to context connection channels, *decoder.conv1x1*.
                * Batch Normalization 2D - Use of batch normalization in convolution block, always used.
                * Activation (Optional) - Use of activation in convolution block, default [ReLU](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html), *decoder.acts*.
                * Dropout (Optional) - Use of dropout in convolution block, *decoder.dropout*.
            * Concatenation of output from Conv1x1 Block and Context Connection Block.
            * Conv3x3 Block - Modifiable depth per block, *decoder.conv3x3*.
                * Convolution - 3x3 convolution from 2*(context connection channels) to output channels, *decoder.conv3x3*.
                * Batch Normalization 2D (Optional) - Use of batch normalization in convolution block, *decoder.norms*.
                * Activation (Optional) - Use of activation in convolution block, default [ReLU](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html), *decoder.acts*.
                * Dropout (Optional) - Use of dropout in convolution block, *decoder.dropout*.
            * Multi Head Cross Attention (Optional per block) - Use of [visual self attention](https://arxiv.org/abs/2010.11929) but between encoder and decoder, please read [U-Net Transformer: Self and Cross Attention for Medical Image Segmentation](https://arxiv.org/abs/2103.06104) for more information under section 2.2. 
                * Patch Embedding - Embedding of patches to be used in self attention, *decoder.embed_dim* and *decoder.patch_size*.
                * Attention - Regular self attention, *decoder.num_heads*, *decoder.embed_dim*.
                * Dropout - Dropout to be used in self attention, *encoder.attn_drop*.