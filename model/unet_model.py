""" Full assembly of the parts to form the complete network """
"""Refer https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py"""

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n = 32

        self.inc = DoubleConv(n_channels, self.n)
        self.down1 = Down(self.n, self.n *2)
        self.down2 = Down(self.n *2, self.n *4)
        self.down3 = Down(self.n *4, self.n *8)
        self.down4 = Down(self.n * 8, self.n * 16)
        factor = 2 if bilinear else 1
        self.down5 = Down(self.n * 16, self.n * 32 //factor)
        self.up1 = (Up(self.n *32, self.n *16 // factor, bilinear))
        self.up2 = (Up(self.n *16, self.n *8 // factor, bilinear))
        self.up3 = (Up(self.n *8, self.n *4 // factor, bilinear))
        self.up4 = (Up(self.n *4, self.n *2 // factor, bilinear))
        self.up5= (Up(self.n *2, self.n, bilinear))
        self.outc = (OutConv(self.n, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)

        logits = self.outc(x)
        return logits

if __name__ == '__main__':
    net = UNet(n_channels=3, n_classes=3)
    print(net)