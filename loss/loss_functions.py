import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from thop import profile, clever_format
# Various loss functions, mainly for segmentation.
# The loss functions are implemented as classes, so that they can be used as a callable object.


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, eps=1e-8):
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, y_pred, y_true):
        y_pred = y_pred.contiguous() # Currently a tensor consiting of K channels of 2D images. Each pixel in channel k is a probability of that pixel belonging to class k.
        y_true = y_true.contiguous() # Currently a tensor consiting of K channels of 2D images. Each channel is a binary mask of class k.
        intersection = (y_pred * y_true).sum(dim=(2, 3))
        union = y_pred.sum(dim=(2, 3)) + y_true.sum(dim=(2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth + self.eps)
        return 1.0 - dice.mean()
    def test_loss(self):
        y_true = T.randint(0,2,(1, 3, 10, 10))
        # y_pred = T.randint(0,2,(1, 3, 10, 10))
        y_pred = T.rand(1, 3, 10, 10)
        # create subplot with 1 row and 2 columns and make the first plot active
        # for k in range(y_true.shape[-3]):
        #     print("Class {}".format(k))
        #     for i in range(y_true.shape[-2]):
        #         for j in range(y_true.shape[-1]):
        #             print(y_true[0,0,i,j].item(),end=" ")
        #         print()
        plt.subplot(1, 2, 1)
        plt.imshow(y_pred[0].detach().numpy().transpose(1,2,0))
        plt.subplot(1, 2, 2)
        print(y_true.shape)
        print(y_true[0,1][None].shape)
        plt.imshow(y_true[0,1][None].detach().numpy().transpose(1,2,0)*255,cmap="gray")
        plt.show()
        return self(y_pred, y_true)
        # Print the y_true as matrix of 0,1,2
class UNetLossFunction(nn.Module):
    def __init__(self, alpha=1.0,beta=1.0,smooth=1.0, eps=1e-8):
        super().__init__()
        self.smooth = smooth
        self.eps = eps
        self.bce = nn.BCELoss()
        self.dice = DiceLoss(smooth, eps)
        self.alpha = alpha
        self.beta = beta

    def forward(self, y_pred, y_true):
        bce = self.bce(y_pred, y_true)
        dice = self.dice(y_pred, y_true)
        return self.alpha*bce + self.beta*dice
    def test_loss(self):
        y_true = T.randint(0,2,(1, 3, 10, 10),dtype=T.float32,requires_grad=True)
        # y_pred = T.randint(0,2,(1, 3, 10, 10))
        y_pred = T.rand(1, 3, 10, 10,dtype=T.float32,requires_grad=True) # .requires_grad_(True)
        return self(y_pred, y_true)
        # Print the y_true as matrix of 0,1,2
if __name__ == "__main__":
    loss = UNetLossFunction()
    l = loss.test_loss()
    l.backward()
    print(loss.test_loss())