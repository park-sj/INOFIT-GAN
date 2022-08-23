#!/usr/bin/python3

import argparse
import os
import glob

from utils import ReplayBuffer
from utils import loadDcm
from utils import saveDcm
from utils import load_file
from os.path import join as ospj
from torch.autograd import Variable
import skimage.transform
import torch

from models import Generator
from datasets import TestDCMDataset
from datasets import get_test_loader

import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='./test', help='root directory of the dataset')
    parser.add_argument('--outroot', type=str, default='./translate_800', help='root directory of the dataset')
    parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
    parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
    parser.add_argument('--cuda', default=True, action='store_true', help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--generator_A2B', type=str, default='output/netG_A2B.pth', help='A2B generator checkpoint file')
    parser.add_argument('--generator_B2A', type=str, default='output/netG_B2A.pth', help='B2A generator checkpoint file')
    parser.add_argument('--target', type=str, default='JW', help='folder name that you want to convert')
    
    opt = parser.parse_args()
    print(opt)
    
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    
    torch.cuda.empty_cache()
    
    ###### Definition of variables ######
    # Networks
    netG_A2B = Generator(opt.input_nc, opt.output_nc)
    netG_B2A = Generator(opt.output_nc, opt.input_nc)
    
    if opt.cuda:
       if torch.cuda.device_count() > 1:
           print("Use", torch.cuda.device_count(), "GPUs")
           netG_A2B = torch.nn.DataParallel(netG_A2B)
           netG_B2A = torch.nn.DataParallel(netG_B2A)
           netG_A2B.cuda()
           netG_B2A.cuda()
    
       elif torch.cuda.device_count() == 1:
           print("Use", torch.cuda.device_count(), "GPUs")
           netG_A2B.cuda()
           netG_B2A.cuda()
    
    # Load state dicts
    netG_A2B.load_state_dict(torch.load(opt.generator_A2B))
    netG_B2A.load_state_dict(torch.load(opt.generator_B2A))
    
    # Set model's test mode
    netG_A2B.eval()
    netG_B2A.eval()
    
    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
    fake_A_buffer = ReplayBuffer()
    
    # Dataset loader
    dataloader = get_test_loader(opt.dataroot, batch_size=opt.batchSize, num_workers=opt.n_cpu)
    
    ###################################
    
    ###### Testing######
    
    # The number of patient
    NP = os.listdir(ospj(opt.dataroot, opt.target))
    
    min_value = -3000.0
    max_value = 2000.0
    
    for _ , patient in enumerate(NP):

        genImage = None
        dcount = 1
        
        dcm = glob.glob(ospj(opt.dataroot, opt.target, patient, '*'))
        dcm = sorted(dcm)
        dcmlen = len(dcm)

        new_header = ospj(opt.outroot, opt.target, patient)
 
        if not os.path.exists(new_header):
            os.makedirs(new_header, exist_ok=True)
        
        for idx, dcmfile in enumerate(dcm):
            
            _, dcmnum = os.path.split(dcmfile)
        
            new_fname = ospj(new_header, dcmnum)
            dcmImage = load_file(str(dcmfile))
            
            dcmImage[dcmImage<min_value] = min_value
            dcmImage[dcmImage>max_value] = max_value
            
            dcmImage = Variable(input_A.copy_(dcmImage))
            
            gen_img = netG_A2B(dcmImage)
            gen_img = gen_img.cpu().detach().numpy().reshape(opt.size, opt.size)
        
            #gen_img[gen_img<min_value] = -750.0
            #gen_img[gen_img>max_value] = 1250.0
            
            if genImage is None:
                genImage = gen_img
                genImage = np.expand_dims(genImage, 0)
                
            else:
                genImage = np.concatenate((genImage, np.expand_dims(gen_img, 0)), axis = 0)
            
            if dcount==dcmlen:
                targetShape, sitkImage, sitkReader = loadDcm(ospj(opt.dataroot, opt.target, patient))
                #genImage = genImage/255 * 2000 - 750
                genImage = skimage.transform.resize(genImage, targetShape)
                saveDcm(genImage, sitkImage, sitkReader, new_header)
                genImage = None
                dcount = 1
                
            dcount = dcount+1
            
        """
        for i, batch in enumerate(dataloader):
            # Set model input
            real_else = Variable(input_A.copy_(batch['else']))
            else_fname = batch['fname']
            
            header, pngname = os.path.split(''.join(else_fname))
        
            dcmlen = len(os.listdir(header))
            
            print(else_fname)
            print(pngname)
            print(len(os.listdir(header)))
            exit()
        
            # Generate output
            fake_else = 0.5 * (netG_A2B(real_else).data + 0.5)
        
            # Save image files
            save_image(fake_A, 'output/A/%04d.png' % (i+1))
            save_image(fake_B, 'output/B/%04d.png' % (i+1))
        
            sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))
        
        sys.stdout.write('\n')
        ##################################
        """