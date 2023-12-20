# -*- coding: utf-8 -*-
"""
Created on Thu June 7 13:24:00 2023

@author: Chih-Wei Tseng
"""

import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class pairDiscriminator(nn.Module):
    def __init__(self):
        super(pairDiscriminator, self).__init__()
        self.negative_slope = 0.03
        self.k = 31
        self.pad = int((self.k-1)/2)
        self.cba_block = nn.Sequential(
            self.C_Vb_A(in_channels=1, out_channels=16, strides=5),
            self.C_Vb_A(in_channels=16, out_channels=32, strides=5),
            self.C_Vb_A(in_channels=32, out_channels=32, strides=5),
            self.C_Vb_A(in_channels=32, out_channels=64, strides=5),
            self.C_Vb_A(in_channels=64, out_channels=128, strides=3)
        )
        self.pool = nn.Sequential(
            nn.Conv1d(128, 1, kernel_size=1, stride=1),
            nn.LeakyReLU(self.negative_slope),
            nn.Linear(in_features=2, out_features=1),
            nn.Sigmoid()
        )


    def C_Vb_A(self, in_channels, out_channels, strides):
        x = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=self.k, stride=strides, padding=self.pad),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(self.negative_slope)
        )
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        x = self.cba_block(x)
        out = self.pool(x)
        return out

class WGAN_pairDiscriminator(nn.Module):
    def __init__(self):
        super(WGAN_pairDiscriminator, self).__init__()
        self.negative_slope = 0.03
        self.k = 31
        self.pad = int((self.k-1)/2)
        self.cba_block = nn.Sequential(
            self.C_Vb_A(in_channels=1, out_channels=16, strides=5),
            self.C_Vb_A(in_channels=16, out_channels=32, strides=5),
            self.C_Vb_A(in_channels=32, out_channels=32, strides=5),
            self.C_Vb_A(in_channels=32, out_channels=64, strides=5),
            self.C_Vb_A(in_channels=64, out_channels=128, strides=3)
        )
        self.pool = nn.Sequential(
            nn.Conv1d(128, 1, kernel_size=1, stride=1),
            nn.LeakyReLU(self.negative_slope),
            nn.Linear(in_features=2, out_features=1)
        )


    def C_Vb_A(self, in_channels, out_channels, strides):
        x = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=self.k, stride=strides, padding=self.pad),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(self.negative_slope)
        )
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        x = self.cba_block(x)
        out = self.pool(x)
        return out


if __name__ == '__main__':
    device = torch.device("cuda", 0)
    model = pairDiscriminator()
    model.to(device)
    
    # summary(model, (1, 1, 3750))
    summary(model, (1, 1, 2500))

