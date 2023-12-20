# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 23:14:00 2022

@author: Chih-Wei Tseng
"""

import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = sigma
        self.noise = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x):
        if self.sigma != 0:
            scale = self.sigma * x.detach()
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x


class SEsubmodule(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super(SEsubmodule, self).__init__()
        pad_size = kernel_size // 2
        
        self.conv = nn.Conv1d(in_channel,out_channel,kernel_size,padding=pad_size,stride=stride)
        self.bn = nn.BatchNorm1d(out_channel)
        self.act = nn.GELU()
        # self.act = nn.GELU()

    def forward(self,x):
        out = self.act(self.bn(self.conv(x)) + x)  #*** 
        return out


class CBAM1D(nn.Module): #channel attention：變形為global avg + global max再合併，後面加上spatial attention
    def __init__(self, nin, nout, kernel_size, stride=1, reduce=4):
        super(CBAM1D, self).__init__()
        self.rb1 = SEsubmodule(nin, nout, kernel_size, stride)
        # self.rb1 = SE_block_module(in_channel=nin, out_channel=nout, kernel_size=kernel_size, stride=stride)
        self.mlp = nn.Sequential(nn.Linear(nout, nout // reduce),
                                nn.GELU(),
                                nn.Linear(nout // reduce, nout))
        self.sigmoid = nn.Sigmoid()
        
        self.conv = nn.Conv1d(2, 1, kernel_size=7, stride=1, padding=3, dilation=1)
    def forward(self, input):
        x = input
        x = self.rb1(x)

        b, c, l = x.size() #batch, channel, signal length
        
        avg_y = F.adaptive_avg_pool1d(x, 1).view(b, c) # global average pooling
        avg_y = self.mlp(avg_y).view(b, c, 1)
        
        max_y = F.adaptive_max_pool1d(x, 1).view(b, c) # global max pooling
        max_y = self.mlp(max_y).view(b, c, 1)
        
        y = self.sigmoid(avg_y + max_y)
        # Mc = y.view(y.size(0), y.size(1), 1)
        # Mf1 = Mc * x #與下面等價
        Mf1 = x * y.expand_as(x)
        
        
        maxOut = torch.max(Mf1, 1)[0].unsqueeze(1)#*** not sure need to check
        avgOut = torch.mean(Mf1, 1).unsqueeze(1)#*** not sure need to check
        Ms = torch.cat((maxOut, avgOut), dim=1)
        
        Ms = self.conv(Ms)
        Ms = self.sigmoid(Ms)
        Ms = Ms.view(Ms.size(0), 1, Ms.size(2))
        Mf2 = Ms * Mf1
        return Mf2
        # out = y + input
        # return out


class GAN_Generator_CBAM(nn.Module):

    def __init__(self):
        super(GAN_Generator_CBAM, self).__init__()

        # -------------------------------Encoder----------------------------------------
        self.enc1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=31, stride=5, padding=15)
        self.att1 = CBAM1D(nin=16, nout=16, kernel_size=31)
        self.enc2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=31, stride=5, padding=15)
        self.att2 = CBAM1D(nin=32, nout=32, kernel_size=31)
        self.enc3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=31, stride=5, padding=15)
        self.att3 = CBAM1D(nin=32, nout=32, kernel_size=31)
        self.enc4 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=31, stride=5, padding=15)
        self.att4 = CBAM1D(nin=64, nout=64, kernel_size=31)
        self.enc5 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=31, stride=3, padding=15)
        self.act = nn.GELU()
        # -------------------------------------------------------------------------------


        # ----------------------------------Decoder-----------------------------------------
        self.dec5 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=31, stride=1, padding=15)
        self.dec4 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=31, stride=3, padding=15, output_padding=2)
        self.dec3 = nn.ConvTranspose1d(in_channels=128, out_channels=32, kernel_size=31, stride=5, padding=15, output_padding=4)
        self.dec2 = nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=31, stride=5, padding=15, output_padding=4)
        self.dec1 = nn.ConvTranspose1d(in_channels=64, out_channels=16, kernel_size=31, stride=5, padding=15, output_padding=4)
        self.final = nn.ConvTranspose1d(in_channels=32, out_channels=1, kernel_size=31, stride=5, padding=15, output_padding=4)

        self.tanh = nn.Tanh()
        self.avePool = nn.AvgPool1d(1, stride=7, padding=0)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.att1(self.act(x1)))
        x3 = self.enc3(self.att2(self.act(x2)))
        x4 = self.enc4(self.att3(self.act(x3)))
        x5 = self.enc5(self.att4(self.act(x4)))
        c = self.act(x5)
        # c = self.act(x5)

        encoded = torch.cat((c, c), dim=1)
        d5 = self.dec5(encoded)
        # d5 = self.dec5(encoded)
        d5_act = self.act(torch.cat((d5, x5), dim=1))
        d4 = self.dec4(d5_act)
        d4_act = self.act(torch.cat((d4, x4), dim=1))
        d3 = self.dec3(d4_act)
        d3_act = self.act(torch.cat((d3, x3), dim=1))
        d2 = self.dec2(d3_act)
        d2_act = self.act(torch.cat((d2, x2), dim=1))
        d1 = self.dec1(d2_act)
        d1_act = self.act(torch.cat((d1, x1), dim=1))
        d0 = self.final(d1_act)
        out = self.tanh(d0)

        return out


class GAN_Generator(nn.Module):
    '''
        reference SeGAN Generator Link:
            https://github.com/leftthomas/SEGAN/blob/master/model.py
        
        input_size = B x 2(channel) x 2520 (84fps * 30s)
        output_size = input_size
    '''
    def __init__(self):
        super(GAN_Generator, self).__init__()

        # -------------------------------Encoder----------------------------------------
        self.enc1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=31, stride=5, padding=15)
        self.enc2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=31, stride=5, padding=15)
        self.enc3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=31, stride=5, padding=15)
        self.enc4 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=31, stride=5, padding=15)
        self.enc5 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=31, stride=3, padding=15)
        self.act = nn.PReLU()
        # -------------------------------------------------------------------------------


        # ----------------------------------Decoder-----------------------------------------
        self.dec5 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=31, stride=1, padding=15)
        self.dec4 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=31, stride=3, padding=15, output_padding=2)
        self.dec3 = nn.ConvTranspose1d(in_channels=128, out_channels=32, kernel_size=31, stride=5, padding=15, output_padding=4)
        self.dec2 = nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=31, stride=5, padding=15, output_padding=4)
        self.dec1 = nn.ConvTranspose1d(in_channels=64, out_channels=16, kernel_size=31, stride=5, padding=15, output_padding=4)
        self.final = nn.ConvTranspose1d(in_channels=32, out_channels=1, kernel_size=31, stride=5, padding=15, output_padding=4)

        self.tanh = nn.Tanh()
        self.avePool = nn.AvgPool1d(1, stride=7, padding=0)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.act(x1))
        x3 = self.enc3(self.act(x2))
        x4 = self.enc4(self.act(x3))
        x5 = self.enc5(self.act(x4))
        c = self.act(x5)
        # c = self.act(x5)

        encoded = torch.cat((c, c), dim=1)
        d5 = self.dec5(encoded)
        # d5 = self.dec5(encoded)
        d5_act = self.act(torch.cat((d5, x5), dim=1))
        d4 = self.dec4(d5_act)
        d4_act = self.act(torch.cat((d4, x4), dim=1))
        d3 = self.dec3(d4_act)
        d3_act = self.act(torch.cat((d3, x3), dim=1))
        d2 = self.dec2(d3_act)
        d2_act = self.act(torch.cat((d2, x2), dim=1))
        d1 = self.dec1(d2_act)
        d1_act = self.act(torch.cat((d1, x1), dim=1))
        d0 = self.final(d1_act)
        out = self.tanh(d0)

        return out


class GAN_Generator_20s_att(nn.Module):
    '''
        reference SeGAN Generator Link:
            https://github.com/leftthomas/SEGAN/blob/master/model.py
        
        input_size = B x 2(channel) x 2520 (84fps * 30s)
        output_size = input_size
    '''
    def __init__(self):
        super(GAN_Generator_20s_att, self).__init__()

        # -------------------------------Encoder----------------------------------------
        self.enc1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=31, stride=5, padding=15)
        self.enc2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=31, stride=5, padding=15)
        self.enc3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=31, stride=5, padding=15)
        self.enc4 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=31, stride=5, padding=15)
        self.act = nn.PReLU()
        # -------------------------------------------------------------------------------


        # ----------------------------------Decoder-----------------------------------------
        self.dec4 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=31, stride=1, padding=15)
        self.dec3 = nn.ConvTranspose1d(in_channels=128, out_channels=32, kernel_size=31, stride=5, padding=15, output_padding=4)
        self.dec2 = nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=31, stride=5, padding=15, output_padding=4)
        self.dec1 = nn.ConvTranspose1d(in_channels=64, out_channels=16, kernel_size=31, stride=5, padding=15, output_padding=4)
        self.final = nn.ConvTranspose1d(in_channels=32, out_channels=1, kernel_size=31, stride=5, padding=15, output_padding=4)

        self.tanh = nn.Tanh()
        self.avePool = nn.AvgPool1d(1, stride=7, padding=0)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.act(x1))
        x3 = self.enc3(self.act(x2))
        x4 = self.enc4(self.act(x3))
        c = self.act(x4)
        # c = self.act(x5)

        encoded = torch.cat((c, c), dim=1)
        d4 = self.dec4(encoded)
        d4_act = self.act(torch.cat((d4, x4), dim=1))
        d3 = self.dec3(d4_act)
        d3_act = self.act(torch.cat((d3, x3), dim=1))
        d2 = self.dec2(d3_act)
        d2_act = self.act(torch.cat((d2, x2), dim=1))
        d1 = self.dec1(d2_act)
        d1_act = self.act(torch.cat((d1, x1), dim=1))
        d0 = self.final(d1_act)
        out = self.tanh(d0)

        return out


class SeGAN_Generator(nn.Module):
    '''
        reference SeGAN Generator Link:
            https://github.com/leftthomas/SEGAN/blob/master/model.py
        
        input_size = B x 2(channel) x 2520 (84fps * 30s)
        output_size = input_size
    '''
    def __init__(self):
        super(SeGAN_Generator, self).__init__()
        self.noise = GaussianNoise(sigma=0.05)
        # -------------------------------Encoder----------------------------------------
        self.enc1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=31, stride=5, padding=15)
        self.enc1_nl = nn.PReLU()
        self.enc2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=31, stride=5, padding=15)
        self.enc2_nl = nn.PReLU()
        self.enc3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=31, stride=5, padding=15)
        self.enc3_nl = nn.PReLU()
        self.enc4 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=31, stride=5, padding=15)
        self.enc4_nl = nn.PReLU()
        # -------------------------------------------------------------------------------


        # ----------------------------------Decoder-----------------------------------------
        self.dec4 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=31, stride=1, padding=15)
        self.dec4_nl = nn.PReLU()
        self.dec3 = nn.ConvTranspose1d(in_channels=128, out_channels=32, kernel_size=31, stride=5, padding=15, output_padding=4)
        self.dec3_nl = nn.PReLU()
        self.dec2 = nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=31, stride=5, padding=15, output_padding=4)
        self.dec2_nl = nn.PReLU()
        self.dec1 = nn.ConvTranspose1d(in_channels=64, out_channels=16, kernel_size=31, stride=5, padding=15, output_padding=4)
        self.dec1_nl = nn.PReLU()
        self.final = nn.ConvTranspose1d(in_channels=32, out_channels=1, kernel_size=31, stride=5, padding=15, output_padding=4)

        self.tanh = nn.Tanh()
        self.avePool = nn.AvgPool1d(1, stride=7, padding=0)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.enc1_nl(x1))
        x3 = self.enc3(self.enc2_nl(x2))
        x4 = self.enc4(self.enc3_nl(x3))
        c = self.enc4_nl(x4)
        # c = self.act(x5)

        z = self.noise(c)
        encoded = torch.cat((c, z), dim=1)
        d4 = self.dec4(encoded)
        d4_act = self.dec4_nl(torch.cat((d4, x4), dim=1))
        d3 = self.dec3(d4_act)
        d3_act = self.dec3_nl(torch.cat((d3, x3), dim=1))
        d2 = self.dec2(d3_act)
        d2_act = self.dec2_nl(torch.cat((d2, x2), dim=1))
        d1 = self.dec1(d2_act)
        d1_act = self.dec1_nl(torch.cat((d1, x1), dim=1))
        d0 = self.final(d1_act)
        out = self.tanh(d0)

        return out

class SeGAN_Generator_CBAM_easy_control(nn.Module):

    def __init__(self, kernel_size=31, dilation=1):
        super(SeGAN_Generator_CBAM_easy_control, self).__init__()
        pad = (dilation*(kernel_size-1))//2
        # -------------------------------Encoder----------------------------------------
        self.enc1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=kernel_size, stride=5, padding=pad, dilation=dilation)
        self.att1 = CBAM1D(nin=16, nout=16, kernel_size=kernel_size)
        self.enc2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=kernel_size, stride=5, padding=pad, dilation=dilation)
        self.att2 = CBAM1D(nin=32, nout=32, kernel_size=kernel_size)
        self.enc3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=kernel_size, stride=5, padding=pad, dilation=dilation)
        self.att3 = CBAM1D(nin=32, nout=32, kernel_size=kernel_size)
        self.enc4 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=kernel_size, stride=5, padding=pad, dilation=dilation)
        self.att4 = CBAM1D(nin=64, nout=64, kernel_size=kernel_size)
        self.enc5 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=kernel_size, stride=3, padding=pad, dilation=dilation)
        self.act = nn.GELU()
        # -------------------------------------------------------------------------------


        # ----------------------------------Decoder-----------------------------------------
        self.dec5 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.dec4 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=kernel_size, stride=3, padding=kernel_size//2, output_padding=2)
        self.dec3 = nn.ConvTranspose1d(in_channels=128, out_channels=32, kernel_size=kernel_size, stride=5, padding=kernel_size//2, output_padding=4)
        self.dec2 = nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=kernel_size, stride=5, padding=kernel_size//2, output_padding=4)
        self.dec1 = nn.ConvTranspose1d(in_channels=64, out_channels=16, kernel_size=kernel_size, stride=5, padding=kernel_size//2, output_padding=4)
        self.final = nn.ConvTranspose1d(in_channels=32, out_channels=1, kernel_size=kernel_size, stride=5, padding=kernel_size//2, output_padding=4)

        self.tanh = nn.Tanh()
        self.avePool = nn.AvgPool1d(1, stride=7, padding=0)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.att1(self.act(x1)))
        x3 = self.enc3(self.att2(self.act(x2)))
        x4 = self.enc4(self.att3(self.act(x3)))
        x5 = self.enc5(self.att4(self.act(x4)))
        c = self.act(x5)
        # c = self.act(x5)

        encoded = torch.cat((c, c), dim=1)
        d5 = self.dec5(encoded)
        # d5 = self.dec5(encoded)
        d5_act = self.act(torch.cat((d5, x5), dim=1))
        d4 = self.dec4(d5_act)
        d4_act = self.act(torch.cat((d4, x4), dim=1))
        d3 = self.dec3(d4_act)
        d3_act = self.act(torch.cat((d3, x3), dim=1))
        d2 = self.dec2(d3_act)
        d2_act = self.act(torch.cat((d2, x2), dim=1))
        d1 = self.dec1(d2_act)
        d1_act = self.act(torch.cat((d1, x1), dim=1))
        d0 = self.final(d1_act)
        out = self.tanh(d0)

        return out

class SeGAN_Generator_CBAM_easy_control_v2(nn.Module):

    def __init__(self, kernel_size=31, dilation=1, downsample=1875):
        super(SeGAN_Generator_CBAM_easy_control_v2, self).__init__()
        pad = (dilation*(kernel_size-1))//2
        # -------------------------------Encoder----------------------------------------
        self.enc1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=kernel_size, stride=5, padding=pad, dilation=dilation)
        self.att1 = CBAM1D(nin=16, nout=16, kernel_size=kernel_size)
        self.enc2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=kernel_size, stride=5, padding=pad, dilation=dilation)
        self.att2 = CBAM1D(nin=32, nout=32, kernel_size=kernel_size)
        self.enc3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=kernel_size, stride=5, padding=pad, dilation=dilation)
        self.att3 = CBAM1D(nin=32, nout=32, kernel_size=kernel_size)
        self.enc4 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=kernel_size, stride=5, padding=pad, dilation=dilation)
        self.att4 = CBAM1D(nin=64, nout=64, kernel_size=kernel_size)
        self.enc5 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=kernel_size, stride=3, padding=pad, dilation=dilation)
        self.act = nn.GELU()
        # -------------------------------------------------------------------------------


        self.get_latent = nn.Sequential( 
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=kernel_size, stride=1, padding=pad, dilation=dilation),
            nn.GELU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=kernel_size, stride=downsample, padding=pad, dilation=dilation)
        )

        # ----------------------------------Decoder-----------------------------------------
        self.dec5 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.dec4 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=kernel_size, stride=3, padding=kernel_size//2, output_padding=2)
        self.dec3 = nn.ConvTranspose1d(in_channels=128, out_channels=32, kernel_size=kernel_size, stride=5, padding=kernel_size//2, output_padding=4)
        self.dec2 = nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=kernel_size, stride=5, padding=kernel_size//2, output_padding=4)
        self.dec1 = nn.ConvTranspose1d(in_channels=64, out_channels=16, kernel_size=kernel_size, stride=5, padding=kernel_size//2, output_padding=4)
        self.final = nn.ConvTranspose1d(in_channels=32, out_channels=1, kernel_size=kernel_size, stride=5, padding=kernel_size//2, output_padding=4)

        self.tanh = nn.Tanh()
        self.avePool = nn.AvgPool1d(1, stride=7, padding=0)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.att1(self.act(x1)))
        x3 = self.enc3(self.att2(self.act(x2)))
        x4 = self.enc4(self.att3(self.act(x3)))
        x5 = self.enc5(self.att4(self.act(x4)))
        c = self.act(x5)
        # c = self.act(x5)

        latent_z = self.get_latent(x)

        encoded = torch.cat((c, latent_z), dim=1)
        d5 = self.dec5(encoded)
        # d5 = self.dec5(encoded)
        d5_act = self.act(torch.cat((d5, x5), dim=1))
        d4 = self.dec4(d5_act)
        d4_act = self.act(torch.cat((d4, x4), dim=1))
        d3 = self.dec3(d4_act)
        d3_act = self.act(torch.cat((d3, x3), dim=1))
        d2 = self.dec2(d3_act)
        d2_act = self.act(torch.cat((d2, x2), dim=1))
        d1 = self.dec1(d2_act)
        d1_act = self.act(torch.cat((d1, x1), dim=1))
        d0 = self.final(d1_act)
        out = self.tanh(d0)

        return out


if __name__ == '__main__':
    device = torch.device("cuda", 0)
    model = SeGAN_Generator()
    model.to(device)
    
    # summary(model, (1, 1, 3750)) # 30*125
    summary(model, (1, 1, 2500)) # 20*125