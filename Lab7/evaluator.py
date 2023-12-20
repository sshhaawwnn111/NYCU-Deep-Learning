import torch
import torch.nn as nn
import torchvision.models as models
from matplotlib import pyplot as plt
from PIL import Image
import os
'''===============================================================
1. Title:     

DLP spring 2023 Lab7 classifier

2. Purpose:

For computing the classification accruacy.

3. Details:

The model is based on ResNet18 with only chaning the
last linear layer. The model is trained on iclevr dataset
with 1 to 5 objects and the resolution is the upsampled 
64x64 images from 32x32 images.

It will capture the top k highest accuracy indexes on generated
images and compare them with ground truth labels.

4. How to use

You should call eval(images, labels) and to get total accuracy.
images shape: (batch_size, 3, 64, 64)
labels shape: (batch_size, 24) where labels are one-hot vectors
e.g. [[1,1,0,...,0],[0,1,1,0,...],...]

==============================================================='''


class evaluation_model():
    def __init__(self):
        #modify the path to your own path
        checkpoint = torch.load('./checkpoint.pth')
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.fc = nn.Sequential(
            nn.Linear(512,24),
            nn.Sigmoid()
        )
        self.resnet18.load_state_dict(checkpoint['model'])
        self.resnet18 = self.resnet18.cuda()
        self.resnet18.eval()
        self.classnum = 24
    def compute_acc(self, out, onehot_labels):
        batch_size = out.size(0)
        acc = 0
        total = 0
        for i in range(batch_size):
            k = int(onehot_labels[i].sum().item())
            total += k
            outv, outi = out[i].topk(k)
            lv, li = onehot_labels[i].topk(k)
            for j in outi:
                if j in li:
                    acc += 1
        return acc / total
    def eval(self, images, labels):
    # def eval(self, images):
        with torch.no_grad():
            #your image shape should be (batch, 3, 64, 64)

            # tmp = images.permute(0, 2, 3, 1)
            # for i in range(3):
            #     plt.imshow(tmp[i].cpu().numpy(), interpolation='nearest')
            #     plt.show()
            # save_image(images, os.path.join(f'./images/check/test.png'), nrow=8)


            

            out = self.resnet18(images)

            # for i in range(8):
            #     print('out', out[i])
            #     print('labels', labels[i])



            acc = self.compute_acc(out.cpu(), labels.cpu())
            return acc