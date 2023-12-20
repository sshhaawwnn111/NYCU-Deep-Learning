import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ReconstructLoss(nn.Module):
    def __init__(self):        
        super(ReconstructLoss, self).__init__()
    def forward(self, outputs, targets):
        
        n_corrcoef = 0
        l1_loss = 0
        
        for i in range(outputs.shape[0]):
            vx = outputs[i] - torch.mean(outputs[i])
            vy = targets[i] - torch.mean(targets[i])
    
            n_corrcoef = n_corrcoef + (1 - (torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))))
            
            l1_loss = l1_loss + F.l1_loss(outputs[i], targets[i])
        
        n_corrcoef = n_corrcoef / outputs.shape[0]
        l1_loss = l1_loss / outputs.shape[0]
        
        loss = l1_loss + n_corrcoef
        
        return loss