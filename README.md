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
                * Dropout - Dropout to be used in self attention, *decoder.attn_drop*.
        * Output Block
            * Conv3x3 Block - 3x3 Convolutional block with softmax activation.

## Data
The data that was used is the ADE20K dataset. This dataset consists of 150 different classes that are as follows:
<details><summary>Class names</summary>

|1           |2         |3        |4             |5               |6            |7            |8          |9             |10            |
|------------|----------|---------|--------------|----------------|-------------|-------------|-----------|--------------|-------------------|
|wall        |building  |sky      |floor         |tree            |ceiling      |road         |bed        |windowpane    |grass              |
|cabinet     |sidewalk  |person   |earth         |door            |table        |mountain     |plant      |curtain       |chair              |
|car         |water     |painting |sofa          |shelf           |house        |sea          |mirror     |rug           |field              |
|armchair    |seat      |fence    |desk          |rock            |wardrobe     |lamp         |bathtub    |railing       |cushion            |
|base        |box       |column   |signboard     |chest of drawers|counter      |sand         |sink       |skyscraper    |fireplace          |
|refrigerator|grandstand|path     |stairs        |runway          |case         |pool table   |pillow     |screen door   |stairway           |
|river       |bridge    |bookcase |blind         |coffee table    |toilet       |flower       |book       |hill          |bench              |
|countertop  |stove     |palm     |kitchen island|computer        |swivel chair |boat         |bar        |arcade machine|hovel              |
|bus         |towel     |light    |truck         |tower           |chandelier   |awning       |streetlight|booth         |television receiver|
|airplane    |dirt track|apparel  |pole          |land            |bannister    |escalator    |ottoman    |bottle        |buffet             |
|poster      |stage     |van      |ship          |fountain        |conveyer belt|canopy       |washer     |plaything     |swimming pool      |
|stool       |barrel    |basket   |waterfall     |tent            |bag          |minibike     |cradle     |oven          |ball               |
|food        |step      |tank     |trade name    |microwave       |pot          |animal       |bicycle    |lake          |dishwasher         |
|screen      |blanket   |sculpture|hood          |sconce          |vase         |traffic light|tray       |ashcan        |fan                |
|pier        |crt screen|plate    |monitor       |bulletin board  |shower       |radiator     |glass      |clock         |flag               |
|UNKNOWN     |          |         |              |                |             |             |           |              |                   |

</details>



