# -*- coding: utf-8 -*-
import torch
from torch import nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import binary_dilation, star


class STE(nn.Module):

    """
    Thresholds during the forward pass and acting as the identity function during the backward pass.
    """
    
    def __init__(self, tr=0.5):
        super(STE, self).__init__()
        self.tr = tr

    def forward(self, input):
        
        out = input.clone()
        out[out > self.tr] = 1
        out[out != 1] = 0

        return out

    def backward(self, input):

        return input


def Dilation(img, dil=1, device=None):
    bs = img.shape[0]
    img = img.detach().cpu().numpy()
    img_dil = np.zeros_like(img)

    for i in range(bs):
        img_dil[i,0] = binary_dilation(img[i,0],selem=star(dil))
    
    return torch.tensor(img_dil).to(device) if device else torch.tensor(img_dil)

    
def visualize_dataset(dataset, img_num=6):
    fig, ax = plt.subplots(2,img_num, figsize = (20,7))
    i = 0 
    for x, y in dataset:
        #print(x.shape, x.float().mean())
        #print(y.shape, y.float().mean())

        image = torchvision.transforms.functional.to_pil_image(x)
        outline = torchvision.transforms.functional.to_pil_image(y)

        ax[0,i].imshow(image)
        ax[1,i].imshow(outline)
        i += 1
        if i == img_num: break
    plt.show()


def add_mask(img, mask):
      img = np.array(img)
      mask = np.array(mask)
      img[:,:,0] = mask*255
      return img

def calc_iou(prediction, ground_truth, tr = None):

    n_images = len(prediction)
    if tr != None:
      for i in range(n_images):
        prediction[i][prediction[i] <= tr] = 0
        prediction[i][prediction[i] > tr] = 1

    intersection, union = 0, 0
    for i in range(n_images):
        intersection += np.logical_and(prediction[i] > 0, ground_truth[i] > 0).astype(np.float32).sum() 
        union += np.logical_or(prediction[i] > 0, ground_truth[i] > 0).astype(np.float32).sum()
    return float(intersection) / union
