import random
import time
import datetime
import sys
import os

from torch.autograd import Variable
import SimpleITK as sitk
import torch
from visdom import Visdom
import numpy as np
import cv2 as cv2

import torchvision.transforms as T
from PIL import Image


def tensor2image(tensor):
    
    image = (tensor[0].cpu().float().numpy()) * 0.5 + 0.5
    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type =cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    return image.astype(np.float32)

class Logger():
    def __init__(self, n_epochs, batches_epoch):
        self.viz = Visdom()
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}


    def log(self, losses=None, images=None):
        #self.mean_period += (time.time() - self.prev_time)
        self.mean_period += np.float32(time.time() - self.prev_time)

        #self.prev_time = time.time()
        self.prev_time = np.float32(time.time())

        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                #self.losses[loss_name] = losses[loss_name].data[0]
                #self.losses[loss_name] = losses[loss_name].data
                self.losses[loss_name] = losses[loss_name].data.cpu().float().numpy()
            else:
                #self.losses[loss_name] += losses[loss_name].data[0]
                #self.losses[loss_name] += losses[loss_name].data
                self.losses[loss_name] += losses[loss_name].data.cpu().float().numpy()

            if (i+1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))

        batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
        batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch

        # Draw images and Plot losses
        for image_name, tensor in images.items():
            if image_name not in self.image_windows:
                self.image_windows[image_name] = self.viz.image(tensor2image(tensor.data), opts={'title':image_name}) 
            else:
                self.viz.image(tensor2image(tensor.data), win=self.image_windows[image_name], opts={'title':image_name})

        if (self.batch % self.batches_epoch) == 0:
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_windows:
                    self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), 
                                                                    opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})                
                else:
                    self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), win=self.loss_windows[loss_name], update='append')

                self.losses[loss_name] = 0.0
            
            self.epoch +=1
            self.batch = 1
            sys.stdout.write('\n')
      
        else:
            self.batch += 1
        
        """
        for loss_name, loss in self.losses.items():
            if loss_name not in self.loss_windows:
                self.loss_windows[loss_name] = self.viz.line(X=np.array([((self.epoch-1)*self.batches_epoch)+self.batch]), Y=np.array([loss/self.batch]), 
                                                                    opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})                
            else:
                self.viz.line(X=np.array([((self.epoch-1)*self.batches_epoch)+self.batch]), Y=np.array([loss/self.batch]), win=self.loss_windows[loss_name], update='append')

            if (self.batch % self.batches_epoch) == 0:
                self.losses[loss_name] = 0.0
                self.epoch += 1
                self.batch = 1
                sys.stdout.write('\n')

        if (self.batch % self.batches_epoch) != 0:
            self.batch += 1
        """


class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

def load_file(filename):
    
    transform = T.Compose([
        T.Resize(256),
        T.ToTensor()
        ])
    
    reader = sitk.ImageFileReader()
    dicomFile = reader.SetFileName(filename)
    image = reader.Execute()
    img = sitk.GetArrayFromImage(image)
    img = img.astype(np.float32).squeeze()
    img = Image.fromarray(img)
    img = transform(img)
    #img = img.squeeze()
    #img = img.transpose((1,2,0))
    return img

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
        os.mkdir(filepath)
        
    img3d = img3d.astype(np.int16)
    newImage = sitk.GetImageFromArray(img3d)

    #newImage.CopyInformation(oldImage)
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