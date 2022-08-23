import cv2
import glob
import random
import os
from itertools import chain
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import albumentations as A
import SimpleITK as sitk
import numpy as np
from torch.utils.data import DataLoader
from random import *

def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
    for ext in ['dcm']]))
    return sorted(fnames)

class TrainDCMDataset(Dataset):
    def __init__(self, root, num_channels=1, transform=None, batch_size=16):
        self.batch_size = batch_size
        self.H_samples, self.H_targets, self.else_samples, self.else_targets = self._make_dataset(root=root)
        self.transform = transform
        self.num_channels = num_channels
        self.normalize = self._normalization()
        self.totensor = self._totensor()

        self.min_value = -750.0
        self.max_value = 1250.0

        #self.H_min_value = -800.0
        #self.H_max_value = 800.0

    def _load_file(self, filename):
        reader = sitk.ImageFileReader()
        dicomFile = reader.SetFileName(filename)
        image = reader.Execute()
        img = sitk.GetArrayFromImage(image)
        #img = img.transpose((1,2,0))
        return img.astype(np.float32).squeeze()

    def _matchlen(self, flen, targetlen, target):
        
        for i in range(flen-targetlen):
            index = randint(0, targetlen-1)
            target.append(target[index])
        
        return target

    def _make_dataset(self, root):

        domains = os.listdir(root)
        all_H_fnames, all_else_fnames, H_domain_labels, else_domain_labels = [], [], [], []
        for idx, domain in enumerate(sorted(domains)):

            if (domain == 'H'):
                H_dir = os.path.join(root, domain)
                H_fnames = listdir(H_dir)
                all_H_fnames += H_fnames
                H_domain_labels += [idx] * len(H_fnames)
            else:
                else_dir = os.path.join(root, domain)
                else_fnames = listdir(else_dir)
                all_else_fnames += else_fnames
                else_domain_labels += [idx] * len(else_fnames)
    
        all_H_fnames = list(all_H_fnames)
        all_else_fnames = list(all_else_fnames)

        H_len = len(all_H_fnames)
        else_len = len(all_else_fnames)

        if(H_len < else_len):
            flen = else_len
            all_H_fnames = self._matchlen(flen, H_len, all_H_fnames)
        else:
            flen = H_len
            all_else_fnames = self._matchlen(flen, else_len, all_else_fnames)

        if(flen % self.batch_size != 0):
            for i in range(self.batch_size - (flen % self.batch_size)):
                index = randint(0, flen-1)
                all_H_fnames.append(all_H_fnames[index])
                all_else_fnames.append(all_else_fnames[index])             

        return all_H_fnames, H_domain_labels, all_else_fnames, else_domain_labels

    def _normalization(self):
        return transforms.Normalize(mean=0.5, std=0.5)

    def _totensor(self):
        return transforms.ToTensor()

    def __getitem__(self, index):
        H_fname = self.H_samples[index % len(self.H_samples)]
        #H_label = self.H_targets[index % len(self.H_targets)]

        else_fname = self.else_samples[index % len(self.else_samples)]
        #else_label = self.else_targets[index % len(self.else_targets)]

        H_img = self._load_file(str(H_fname))
        else_img = self._load_file(str(else_fname))
        
        H_img[H_img>self.H_max_value] = self.H_max_value
        H_img[H_img<self.H_min_value] = self.H_min_value
        
        #else_img[else_img>self.max_value] = self.max_value
        #else_img[else_img<self.min_value] = self.min_value

        transform_bundle = np.concatenate((np.expand_dims(H_img, 2), np.expand_dims(else_img, 2)), axis = 2)
        transform_result = self.transform(image=transform_bundle)['image']
        H_img = transform_result[:,:,0]
        else_img = transform_result[:,:,1]
        
        #H_img = self.transform(image=np.expand_dims(H_img, 2))['image']
        #else_img = self.transform(image=np.expand_dims(else_img, 2))['image']
        
        H_img = self.totensor(H_img)
        H_img = self.normalize(H_img)
        else_img = self.totensor(else_img)
        else_img = self.normalize(else_img)
        
        return {'else': else_img, 'H': H_img}

    def __len__(self):
        return max( len(self.H_samples), len(self.else_samples) )

class TestDCMDataset(Dataset):
    def __init__(self, root, num_channels=1, transform=None, batch_size=16):
        self.batch_size = batch_size
        self.else_samples, self.else_targets = self._make_dataset(root=root)
        self.transform = transform
        self.num_channels = num_channels
        self.normalize = self._normalization()
        self.totensor = self._totensor()

    def _load_file(self, filename):
        reader = sitk.ImageFileReader()
        dicomFile = reader.SetFileName(filename)
        image = reader.Execute()
        img = sitk.GetArrayFromImage(image)
        #img = img.transpose((1,2,0))
        return img.astype(np.float32).squeeze()

    def _make_dataset(self, root):

        domains = os.listdir(root)
        all_else_fnames, else_domain_labels = [], []

        for idx, domain in enumerate(sorted(domains)):
            else_dir = os.path.join(root, domain)
            else_fnames = listdir(else_dir)
            all_else_fnames += else_fnames
            else_domain_labels += [idx] * len(else_fnames)
    
        all_else_fnames = list(all_else_fnames)
        
        return all_else_fnames, else_domain_labels

    def _normalization(self):
        return transforms.Normalize(mean=0.5, std=0.5)

    def _totensor(self):
        return transforms.ToTensor()

    def __getitem__(self, index):
        else_fname = self.else_samples[index % len(self.else_samples)]
        #else_label = self.else_targets[index % len(self.else_targets)]
        else_img = self._load_file(str(else_fname))
        else_img = self.transform(image=else_img)['image']
            
        return {'else': else_img, 'fname': str(else_fname)}

    def __len__(self):
        return len(self.else_samples)
"""
def get_transform(train=True, img_size=256, prob=0.5, num_channels=3):

    if train:
        transform_list = [
            A.Resize(int(img_size*1.12), int(img_size*1.12), Image.BICUBIC),
            A.RandomCrop(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.Resize(img_size, img_size)
        ]
    else:
        transform_list = [
            A.Resize(img_size, img_size),
        ]

    transform = A.Compose(transform_list)
    return transform
"""

def get_transform(train=True, img_size=256, prob=0.5, num_channels=3):
    if train:
        transform_list = [
            A.RandomResizedCrop(img_size, img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1), interpolation=1, always_apply=True, p=prob),
            A.HorizontalFlip(p=0.5),
            A.Resize(img_size, img_size)
        ]
    else:
        transform_list = [
            A.Resize(img_size, img_size),
        ]

    transform = A.Compose(transform_list)

    return transform

def get_train_loader(root, img_size=256, batch_size=16, shuffle=True, num_workers=8, **kwargs):

    print('Preparing train DataLoader for the generation phase...')

    transform = get_transform(train=True, img_size=img_size, num_channels=1)

    dataset = TrainDCMDataset(root, num_channels=1, transform=transform)

    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=None, pin_memory=True)

def get_test_loader(root, img_size=256, batch_size=1, shuffle=False, num_workers=8, **kwargs):

    print('Preparing test DataLoader for the generation phase...')

    transform = get_transform(train=False, img_size=img_size, num_channels=1)

    dataset = TestDCMDataset(root, num_channels=1, transform=transform)

    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=None, pin_memory=True)
