# -*- coding: utf-8 -*-
"""Discriminator.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pinFRdoCaOOkILvErWu3VJlnfpIhmuEP
"""

import torch
from torch import nn
from TopoAL.src.DiscriminatorLabels import generate_labels

class ResConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, leaky_alpha=0.02):
        super(ResConv2d, self).__init__()
        self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.layer_block  = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, padding=self.padding),
                                          nn.LeakyReLU(leaky_alpha, inplace=False))

    def forward(self, x):
        return self.layer_block(x)

class ResBlock(nn.Module):

    def __init__(self, n_filters, num_layers=1, leaky_alpha=0.02):
        super(ResBlock, self).__init__()

        self.kernel_sizes = [(3,1) for i in range(num_layers)]
        layers = [ResConv2d(n_filters, n_filters, self.kernel_sizes[i], leaky_alpha) for i in range(num_layers)]
        self.res_block = nn.Sequential(*layers)
    
    def forward(self, input):
      
        out = self.res_block(input)
        out += input

        return out

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=1):
        super(DiscriminatorBlock, self).__init__()

        self.conv_strided = nn.Conv2d(in_channels, out_channels, kernel_size=(3,1), stride=2, padding=(1,0))
        self.ResBlock = ResBlock(out_channels,num_layers=num_layers)

    def forward(self, input):

        out = self.conv_strided(input)
        out = self.ResBlock(out)

        return out


class Discriminator(nn.Module):
    def __init__(self, cn=64, in_channels = 3, num_layers=1):
        super(Discriminator, self).__init__()

        
        self.stage_1 = nn.Sequential(DiscriminatorBlock(in_channels, cn, num_layers=num_layers),
                                    DiscriminatorBlock(cn, cn*2, num_layers=num_layers),
                                    DiscriminatorBlock(cn*2, cn*4, num_layers=num_layers),
                                    DiscriminatorBlock(cn*4, cn*8, num_layers=num_layers),
                                    DiscriminatorBlock(cn*8, cn*8, num_layers=num_layers))
        self.conv_1 = nn.Conv2d(cn*8, 1, kernel_size=(1,1))

        self.stage_2 = DiscriminatorBlock(cn*8, cn*8, num_layers=num_layers)
        self.conv_2 = nn.Conv2d(cn*8, 1, kernel_size=(1,1))

        self.stage_3 = DiscriminatorBlock(cn*8, cn*8, num_layers=num_layers)
        self.conv_3 = nn.Conv2d(cn*8, 1, kernel_size=(1,1))

        self.stage_4 = DiscriminatorBlock(cn*8, cn*8, num_layers=num_layers)
        self.conv_4 = nn.Conv2d(cn*8, 1, kernel_size=(1,1))


    def assign_labels(self, mask_gt, mask_p, sigma=0.5, tr=0.5):

        bs = mask_p.shape[0]
        labels1 = torch.rand((bs,1,8,8))
        labels2 = torch.rand((bs,1,4,4))
        labels3 = torch.rand((bs,1,2,2))
        labels4 = torch.rand((bs,1,1,1))

        labels = []
        for i in range(bs):
            labels  = generate_labels(mask_gt[i,0].numpy(), mask_p[i,0].numpy(), sigma=sigma, tr=tr)
            labels1[i,0] = labels[0]
            labels2[i,0] = labels[1]
            labels3[i,0] = labels[2]
            labels4[i,0] = labels[3]

        return [labels1, labels2, labels3, labels4] 


    def forward(self, input):
        
        out = self.stage_1(input)

        out_1 = self.conv_1(out)

        out = self.stage_2(out)
        out_2 = self.conv_2(out)

        out = self.stage_3(out)
        out_3 = self.conv_3(out)

        out = self.stage_4(out)
        out_4 = self.conv_4(out)

        return [out_1.sigmoid(), out_2.sigmoid(), out_3.sigmoid(), out_4.sigmoid()]