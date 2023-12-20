import glob
import os
import json
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image
from evaluator import evaluation_model
import torchvision.transforms as transforms
import numpy as np

def one_hot(a, num_classes):
    return torch.squeeze(np.eye(num_classes)[a.reshape(-1)])
def sample_timesteps(n):
        return one_hot(torch.randint(low=0, high=300 - 1, size=(n,)), 300)


t = (torch.ones(8) * 299).long()

time = one_hot(t, 300)
# time = time.unsqueeze(-1)

print(time.shape)