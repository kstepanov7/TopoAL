# -*- coding: utf-8 -*-
"""DataWrappers.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16Hj8vFFsq-8AzTCetKDhZqcbg_1_qYo_
"""

import os
import random
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, ConcatDataset, DataLoader

from skimage.morphology import dilation, star
from PIL import Image, ImageOps

import matplotlib.pyplot as plt
import glob

class RandomDataset(Dataset):
    
    """
      Generate Dataset from randomly cropped samples from given images
      
      Args:
            images_dir: path to directory with images
            masks_dir: path to directory with masks
            img_transform, mask_transform: function for sample transformation
            sample_size: spatial resolution of sample in pixels (height, width)
            dil: structuring element for mask dilation
    """

    def __init__(self, images_dir, masks_dir, img_transform,mask_transform, test=False, 
                 sample_size=(1024,1024), dil=star(0), inv=False, preprocessing_fn=None):

        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.images_titles = sorted(os.listdir(self.images_dir))
        self.masks_titles = sorted(os.listdir(self.masks_dir))
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.preprocessing_fn = preprocessing_fn
        self.sample_size = sample_size
        self.test = test
        self.dil = dil
        self.inv = inv
        
    def __len__(self):
        return len(self.images_titles)

    def add_border(self, img):
        old_size = img.size

        new_size = ((int(old_size[0] / self.sample_size[0]) + 1)*self.sample_size[0], 
                    (int(old_size[1] / self.sample_size[1]) + 1)*self.sample_size[1])
        
        new_img = Image.new(img.mode, new_size, 0)
        new_img.paste(img, (int((new_size[0]-old_size[0])/2),
                            int((new_size[1]-old_size[1])/2)))
        
        return new_img

    def __getitem__(self, idx):

        image = Image.open(os.path.join(self.images_dir, self.images_titles[idx]))
        mask = Image.open(os.path.join(self.masks_dir, self.masks_titles[idx]))
        if self.test:
            mask = dilation(mask, self.dil)
            return image, mask

        if self.inv:
          image = image.resize((int(image.size[0]/6), int(image.size[1] /6)))
          mask = mask.resize((int(mask.size[0]/6), int(mask.size[1] /6)))
          mask = ImageOps.invert(mask)

        h, w = self.sample_size
        if image.size[0] < h or image.size[1] < w:
            image = self.add_border(image)
            mask = self.add_border(mask)
        
        x_range = max(1, image.size[0] - w)
        y_range = max(1, image.size[1] - h)

        x = np.random.randint(x_range)
        y = np.random.randint(y_range)

        image = image.crop((x, y, x+h, y+w))
        mask = mask.crop((x, y, x+h, y+w))
          
        
        seed = random.randint(0, 100)
        random.seed(seed)
        torch.manual_seed(seed)
        sample = self.img_transform(image=np.array(image), mask=np.array(mask))
        image, mask = sample['image'], sample['mask']
        if len(mask.shape) == 3:
          mask = mask[:,:,1]

        mask = dilation(mask, self.dil)

        #random.seed(seed)
        #torch.manual_seed(seed)
        #mask = self.mask_transform(image=np.array(mask,dtype=np.uint8))['image']
        mask[mask != 0] = 1
        if self.preprocessing_fn:
            image = self.preprocessing_fn(image)
        image = np.transpose(image, (2,0,1))
        mask = mask.reshape(1,mask.shape[0],mask.shape[1])

        return torch.tensor(image).float(), torch.tensor(mask)

class ValDataset(Dataset):

    def __init__(self, data_dir, sample_size=(256,256), transform=None, inv=False, preprocess_fn=None):

        self.transform = transform
        self.inv = inv
        self.preprocess_fn = preprocess_fn
        self.sample_size = sample_size
        self.images, self.masks = self.create_samples(data_dir)

    def __len__(self):
        return len(self.images)

    def add_border(self, img):
        old_size = img.size

        new_size = ((int(old_size[0] / self.sample_size[0]) + 1)*self.sample_size[0], 
                    (int(old_size[1] / self.sample_size[1]) + 1)*self.sample_size[1])
            
        new_img = Image.new(img.mode, new_size, 0)
        new_img.paste(img, (int((new_size[0]-old_size[0])/2),
                            int((new_size[1]-old_size[1])/2)))
            
        return new_img

    def crop_samples(self, img):
        
        x_samples = img.size[0]//self.sample_size[0]
        y_samples = img.size[1]//self.sample_size[1]

        samples = []
        ss = self.sample_size
        for i in range(x_samples):
            for j in range(y_samples):
                sample = np.array(img.crop((i*ss[0], j*ss[0], (i+1)*ss[0], (j+1)*ss[1])))
                samples.append(sample)

        return samples

    def create_samples(self, data_dir):

        img_paths = glob.glob(data_dir+'/images/*.tif')
        mask_paths = glob.glob(data_dir+'/masks/*.png')

        images, masks = [], []
        for img_path, mask_path in zip(img_paths, mask_paths):
            img = Image.open(img_path)
            mask = Image.open(mask_path)

            if self.inv:
                img = img.resize((int(img.size[0]/6), int(img.size[1] /6)))
                mask = mask.resize((int(mask.size[0]/6), int(mask.size[1] /6)))
                mask = ImageOps.invert(mask)

            img = self.add_border(img)
            mask = self.add_border(mask)

            img_s = self.crop_samples(img)
            mask_s = self.crop_samples(mask)

            images.extend(img_s)
            masks.extend(mask_s)

        return images, masks

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]
        if self.transform:
          
            seed = 42
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

            sample = self.transform(image=np.array(img), mask=np.array(mask))
            img, mask = sample['image'], sample['mask']

        if self.preprocess_fn:
            img = self.preprocess_fn(img)
        if len(mask.shape) == 3:
            mask = mask[:,:,1]

        mask[mask > 0] = 1
        img = img.transpose(2,0,1)
        mask = mask.reshape(1,mask.shape[0],mask.shape[1])
        return torch.tensor(img).float(), torch.tensor(mask).float()
    
    
def create_dataloaders(OTT = True, MOS = False, URBN = False,transforms=None, sample_size = (256,256), prep_fn=None, bs=4, dil=None):

    train_datasets, val_datasets = [], []
    if MOS:
        DATADIR = '/content/drive/My Drive/GeoAlert/TAT+MOS/MOS'
        train_datasets.append(RandomDataset(f'{DATADIR}/train/images/',f'{DATADIR}/train/masks/', 
                                    transforms, None,sample_size=sample_size, preprocessing_fn=prep_fn, dil=dil))
        val_datasets.append(RandomDataset(f'{DATADIR}/val/images/',f'{DATADIR}/val/masks/', 
                                    transforms, None,sample_size=sample_size, preprocessing_fn=prep_fn, dil=dil))

    if OTT:
        DATADIR = '/content/drive/My Drive/GeoAlert/Ottawa-Dataset'
        train_datasets.append(RandomDataset(f'{DATADIR}/train/images/',f'{DATADIR}/train/masks/', 
                                    transforms, None,sample_size=(256,256), inv=True, preprocessing_fn=prep_fn, dil=dil))
        val_datasets.append(RandomDataset(f'{DATADIR}/val/images/',f'{DATADIR}/val/masks/', 
                                    transforms, None,sample_size=(256,256), inv=True, preprocessing_fn=prep_fn, dil=dil))

    train_loader = DataLoader(torch.utils.data.ConcatDataset(train_datasets), batch_size=bs, shuffle=True)
    val_loader = DataLoader(torch.utils.data.ConcatDataset(val_datasets), batch_size=bs, shuffle=True)

    return train_loader, val_loader

