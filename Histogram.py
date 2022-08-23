# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import os

def thresholdImage(imageArr):
    imageArr[imageArr<-750]=-750
    imageArr[imageArr>1250]=1250
    image = sitk.GetImageFromArray(imageArr)
    image = sitk.Cast(image, sitk.sitkInt16)
    return image, imageArr
def readDicom(dir):
    fileReader = sitk.ImageSeriesReader()
    dicomFiles = fileReader.GetGDCMSeriesFileNames(dir)
    fileReader.SetFileNames(dicomFiles)
    fileReader.MetaDataDictionaryArrayUpdateOn()
    fileReader.LoadPrivateTagsOn()
    image = fileReader.Execute()
    image = sitk.Cast(image, sitk.sitkInt16)
    imageArr = sitk.GetArrayFromImage(image)
    return image, imageArr, fileReader
    
dirUserNames = os.listdir("./Match_source/JW")

for dirUserName in dirUserNames:
#     if 'H_LeeTaeGyeong' in dirUserName:
    directory = os.path.join("./Match_source/JW", dirUserName)
    image, imageArr, imageReader = readDicom(directory)
    #image, imageArr = thresholdImage(imageArr)
#     plt.hist(np.array(imageArr).ravel(), alpha=0.9, range=(-2000, 1000), bins=100, log=True, color='b')
    plt.hist(np.array(imageArr).ravel(), alpha=0.9, range=(-5000, 5000), bins=100, log=False, color='b')
    plt.title(dirUserName)
    plt.xlabel('intensity')
    plt.ylabel('frequency')
    plt.show
    plt.savefig('./' + dirUserName + 'Histo.png')
    plt.clf()