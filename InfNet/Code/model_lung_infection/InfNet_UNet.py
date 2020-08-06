# -*- coding: utf-8 -*-

"""Preview
Code for 'Inf-Net: Automatic COVID-19 Lung Infection Segmentation from CT Scans'
submit to Transactions on Medical Imaging, 2020.

First Version: Created on 2020-05-05 (@author: Ge-Peng Ji)
Second Version: Fix some bugs and edit some parameters on 2020-05-15. (@author: Ge-Peng Ji)
"""

import torch.nn.functional as F
from InfNet.Code.model_lung_infection.module.unet_parts import *


class Inf_Net_UNet(nn.Module):
    """ Full assembly of the parts to form the complete network """
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(Inf_Net_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.out_inpainting = OutConv(64, n_channels)
        self.outc = OutConv(64, n_classes)

    def forward_inpainting(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        reconstructed_image = F.tanh(self.out_inpainting(x))

        return reconstructed_image

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits


class Inf_Net_UNet_Improved(nn.Module):
    """ Full assembly of the parts to form the complete network """
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(Inf_Net_UNet_Improved, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(262, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1030, 512 // factor, bilinear)
        self.up2 = Up(518, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        # inpainting weights
        self.up1_inpainting = Up(1030, 512 // factor, bilinear)
        self.up2_inpainting = Up(518, 256 // factor, bilinear)
        self.up3_inpainting = Up(256, 128 // factor, bilinear)
        self.up4_inpainting = Up(128, 64, bilinear)
        self.out_inpainting = OutConv(64, n_channels)

    def forward_inpainting(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x3 = torch.cat((x3, F.interpolate(x, size=32)), 1)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = torch.cat((x5, F.interpolate(x, size=8)), dim=1)
        x_out = self.up1_inpainting(x5, x4)
        x_out = self.up2_inpainting(x_out, x3)
        x_out = self.up3_inpainting(x_out, x2)
        x_out = self.up4_inpainting(x_out, x1)
        reconstructed_image = F.tanh(self.out_inpainting(x_out))

        return reconstructed_image

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x3 = torch.cat((x3, F.interpolate(x, size=32)), dim=1)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = torch.cat((x5, F.interpolate(x, size=8)), dim=1)
        x_out = self.up1(x5, x4)
        x_out = self.up2(x_out, x3)
        x_out = self.up3(x_out, x2)
        x_out = self.up4(x_out, x1)
        logits = self.outc(x_out)
        return logits
