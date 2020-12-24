# -*- coding: utf-8 -*-


import torch
from torch import nn

class Loss_D(nn.Module):
    
    def __init__(self, loss = nn.BCELoss()):
        super(Loss_D, self).__init__()
        self.Loss = loss

    def forward(self, pred_D, labels_D, pred_xy):

        loss = 0 
        for i in range(len(pred_D)):

            loss += self.Loss(pred_D[i], labels_D[i].float()) + torch.mean(torch.sum(-torch.log(pred_xy[i]), (2,3)))
            #loss += (self.BCELoss(pred_D[i].sigmoid(), labels_D[i]))
            
        return loss

class Loss_G(nn.Module):
    def __init__(self, loss = nn.BCELoss(), l = 0.005):
        super(Loss_G, self).__init__()
        self.Loss = loss
        self.l = l

    def forward(self, mask_gt, mask_pred, pred_D):

        loss = self.Loss(mask_pred, mask_gt) 
        for i in range(len(pred_D)):

            loss = loss + self.l * torch.mean(torch.sum(-torch.log(pred_D[i]), (2,3)))
        return loss
