import os
import glob
from os.path import join as ospj
import numpy as np
import time
import datetime
from itertools import chain
from pathlib import Path
import torch
import torch.nn as nn
import torchvision
import torchvision.utils as vutils
import torch.nn.functional as F
import cv2

from PIL import Image

import SimpleITK as sitk
import skimage.transform

def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
    for ext in ['dcm']]))
    return sorted(fnames)

def loadDcm(dir):
    assert os.path.isdir(dir), f'Cannot find the directory {dir}'
    reader = sitk.ImageSeriesReader()
    dicomFiles = reader.GetGDCMSeriesFileNames(dir)
    reader.SetFileNames(dicomFiles)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    image = reader.Execute()
    img3d = sitk.GetArrayFromImage(image)
    return img3d.shape, image, reader

def saveDcm(img3d, oldImage, reader, filepath):
    if not os.path.isdir(filepath):
        os.makedirs(filepath)
        
    img3d = img3d.astype(np.int16)
    newImage = sitk.GetImageFromArray(img3d)

    newImage.CopyInformation(oldImage)
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    sp_x, sp_y = reader.GetMetaData(0, "0028|0030").split('\\')
    _, _, z_0 = reader.GetMetaData(0, "0020|0032").split('\\')
    _, _, z_1 = reader.GetMetaData(1, "0020|0032").split('\\')
    spacing_ratio = np.array([1, 1, 1], dtype=np.float64)
    sp_z = abs(float(z_0) - float(z_1))
    sp_z = float(sp_z) / spacing_ratio[0]
    sp_x = float(sp_x) / spacing_ratio[1]
    sp_y = float(sp_y) / spacing_ratio[2]
    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")
    direction = newImage.GetDirection()
    series_tag_values = [(k, reader.GetMetaData(0, k)) for k in reader.GetMetaDataKeys(0)] + \
                         [("0008|0031", modification_time),
                         ("0008|0021", modification_date),
                         ("0028|0100", "16"),
                         ("0028|0101", "16"),
                         ("0028|0102", "15"),
                         ("0028|0103", "1"),
                         ("0028|0002", "1"),
                         ("0008|0008", "DERIVED\\SECONDARY"),
                         ("0020|000e", "1.2.826.0.1.3680043.2.1125." + modification_date + ".1" + modification_time),
                         ("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6], direction[1], direction[4], direction[7]))))]
    tags_to_skip = ['0010|0010', '0028|0030', '7fe0|0010', '7fe0|0000', '0028|1052',
                    '0028|1053', '0028|1054', '0010|4000', '0008|1030', '0010|1001',
                    '0008|0080', '0010|0040']
    for i in range(newImage.GetDepth()):
        image_slice = newImage[:, :, i]
        # image_slice.CopyInformation(oldImage[:, :, i])
        for tag, value in series_tag_values:
            if (tag in tags_to_skip):
                continue
            image_slice.SetMetaData(tag, value)
        image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))
        image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))
        image_slice.SetMetaData('0020|0032', reader.GetMetaData(i, "0020|0032"))
        image_slice.SetMetaData("0020|0013", str(i))
        image_slice.SetMetaData('0028|0030', '\\'.join(map(str, [sp_x, sp_y])))
        image_slice.SetSpacing([sp_x, sp_y])
        image_slice.SetMetaData("0018|0050", str(sp_z))
        writer.SetFileName(os.path.join(filepath, str(i).zfill(3) + '.dcm'))
        writer.Execute(image_slice)

def make_noise(std, gray):
    width, height = gray.shape
    img_noise = np.zeros((width, height), dtype=np.float32)

    noise = std * np.random.normal(size=(width, height))
    img_noise = gray + noise

    """
    for h in range(height):
        for w in range(width):
            noise = np.random.normal()
            set_noise = std * noise
            img_noise[h][w] = gray[h][w] + set_noise
    """

    return img_noise

class DCMResolution():
    def __init__(self, root, result, num_channels=1):
        self._make_dataset(root, result)

        #self.min_value = -750.0
        #self.max_value = 1250.0

    def _load_file(self, filename):
        reader = sitk.ImageFileReader()
        dicomFile = reader.SetFileName(filename)
        image = reader.Execute()
        img = sitk.GetArrayFromImage(image)
        #img = img.transpose((1,2,0))
        return img.astype(np.float32).squeeze()

    def _make_dataset(self, root, result):
        domains = os.listdir(root)

        domain_labels, patient_num, all_fnames  = [], [], []
        imgbundle = None
        std = 100

        print('Start Matching...')

        if not os.path.exists(result):
            os.makedirs(result)
            print("The result directory does not exist, so that will be created")

        for idx, domain in enumerate(sorted(domains)):
            dir_name = os.path.join(root, domain)
            patient_num = os.listdir(dir_name)

            for patient in patient_num:
                kratio = 40
                fnames= listdir(os.path.join(root, domain, patient))
                result_path = os.path.join(result, domain, patient)

                for fname in fnames:
                    img = self._load_file(str(fname))
                    ksize = int(img.shape[0] / kratio)

                    if ksize % 2 == 0:
                        img = cv2.GaussianBlur(img, (ksize+1, ksize+1), 0)
                    else:
                        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

                    img = make_noise(std, img)

                    if imgbundle is None:
                        imgbundle = np.expand_dims(img, 0)
                    else:
                        imgbundle = np.concatenate((imgbundle, np.expand_dims(img, 0)), axis = 0)

                targetShape, sitkImage, sitkReader= loadDcm(os.path.join(root, domain, patient))
                saveDcm(imgbundle, sitkImage, sitkReader, result_path)
                
                imgbundle = None

                print('Matching at {}'.format(result_path))

        print("finished Matching")

if __name__ == '__main__':
    DCMResolution(root="./Match_source", result="./Match_result")
    
    
