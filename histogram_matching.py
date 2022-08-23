# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import os
import torch
import time
from skimage.filters import laplace
from skimage.transform import downscale_local_mean, resize
import skimage.transform

def readDicom(dir):
    fileReader = sitk.ImageSeriesReader()
    dicomFiles = fileReader.GetGDCMSeriesFileNames(dir)
    fileReader.SetFileNames(dicomFiles)
    fileReader.MetaDataDictionaryArrayUpdateOn()
    fileReader.LoadPrivateTagsOn()
    image = fileReader.Execute()
#     image = sitk.Cast(image, sitk.sitkInt16)
    imageArr = sitk.GetArrayFromImage(image)
    return image, imageArr, fileReader
    
def saveDicom(newArray, filepath, oldImage, reader):
    if not os.path.isdir(filepath):
        os.mkdir(filepath)
    newArray = np.squeeze(newArray)
    intArray = newArray.astype(np.int16)
#     newArray = skimage.transform.resize(newArray, oldImageArr.shape, anti_aliasing=False)
#     newArray[newArray > 0.5] = 1
#     newArray[newArray <= 0.5] = 0
#     paddedArray = newArray.astype(np.int16)
    newImage = sitk.GetImageFromArray(intArray)
#     newImage = sitk.Cast(newImage, sitk.sitkInt16)
    newImage.CopyInformation(oldImage)
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    spacing_ratio = np.array([1, 1, 1], dtype=np.float64)
    sp_x, sp_y = reader.GetMetaData(0, "0028|0030").split('\\')
    _, _, z_0 = reader.GetMetaData(0, "0020|0032").split('\\')
    _, _, z_1 = reader.GetMetaData(1, "0020|0032").split('\\')
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
                         ("0028|0010", "296"),
                         ("0028|0011", "296"),
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
        for tag, value in series_tag_values:
            if (tag in tags_to_skip):
                continue
            image_slice.SetMetaData(tag, value)
        image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))
        image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))
        image_slice.SetMetaData('0020|0032', '\\'.join(map(str, [0, 0, i * sp_z])))
        image_slice.SetMetaData("0020|0013", str(i))
        image_slice.SetMetaData('0028|0030', '\\'.join(map(str, [sp_x, sp_y])))
        image_slice.SetSpacing([sp_x, sp_y])
        image_slice.SetMetaData("0018|0050", str(sp_z))
        writer.SetFileName(os.path.join(filepath, str(i).zfill(3) + '.dcm'))
        writer.Execute(image_slice)
# def saveDicom(newImage, filepath, oldImage, reader):
#     writer = sitk.ImageFileWriter()
#     writer.KeepOriginalImageUIDOn()
#     modification_time = time.strftime("%H%M%S")
#     modification_date = time.strftime("%Y%m%d")
#     direction = newImage.GetDirection()
#     series_tag_values = [(k, reader.GetMetaData(0, k)) for k in reader.GetMetaDataKeys(0)] + \
#                         [("0008|0031", modification_time),
#                         ("0008|0021", modification_date),
#                         ("0008|0008", "DERIVED\\SECONDARY"),
#                         ("0020|000e", "1.2.826.0.1.3680043.2.1125." + modification_date + ".1" + modification_time),
#                         ("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6], direction[1], direction[4], direction[7]))))]
#     for i in range(newImage.GetDepth()):
#         image_slice = newImage[:, :, i]
#         image_slice.CopyInformation(oldImage[:, :, i])
#         for tag, value in series_tag_values:
#             if (tag == '0010|0010'):
#                 continue
#             image_slice.SetMetaData(tag, value)
#         image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))
#         image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))
#         image_slice.SetMetaData('0020|0032', '\\'.join(map(str, newImage.TransformIndexToPhysicalPoint((0, 0, i)))))
#         image_slice.SetMetaData("0020|0013", str(i))
#         writer.SetFileName(os.path.join(filepath, str(i).zfill(3) + '.dcm'))
#         writer.Execute(image_slice)
        
#JW = -3000, 2000
#H = -1200, 800

def thresholdImage(imageArr, H_name):
    
    if (H_name == 'H'):
        Hmin = -1200
        Hmax = 800
        
    elif (H_name == 'JW'):
        Hmin = -3000
        Hmax = 2000
        
    imageArr[imageArr<Hmin] = Hmin
    imageArr[imageArr>Hmax] = Hmax
    image = sitk.GetImageFromArray(imageArr)
#     image = sitk.Cast(image, sitk.sitkInt16)
    return image, imageArr
    
# JW병원 히스토그램 평균 구하기 
dirH_UserNames = os.listdir("./Match_source/JW")
image, imageArr, reader = readDicom("./Match_source/JW/JW_14")
image, imageArr = thresholdImage(imageArr, 'JW')
imageArr = resize(imageArr.astype(np.float32), (600, 600, 600), anti_aliasing = False)
#imageHisAverageTuple = plt.hist(np.array(imageArr).ravel(), alpha=0.5, range=(-750, 1250), bins=2000, log=False, color='r')
imageHisAverageTuple = plt.hist(np.array(imageArr).ravel(), alpha=0.5, range=(-3000, 2000), bins=5000, log=False, color='r')
imageHisAverageList = list(imageHisAverageTuple)
numberOfH_CT = 1

for dirH_UserName in dirH_UserNames:
    if('JW_14' in dirH_UserName):
        continue
    image, imageArr, reader = readDicom('./Match_source/JW/' + dirH_UserName)
    image, imageArr = thresholdImage(imageArr, 'JW')
    imageArr = resize(imageArr.astype(np.float32), (600, 600, 600), anti_aliasing = False)
    #imageHisTuple = plt.hist(np.array(imageArr).ravel(), alpha=0.5, range=(-750, 1250), bins=2000, log=False, color='r')
    imageHisTuple = plt.hist(np.array(imageArr).ravel(), alpha=0.5, range=(-3000, 2000), bins=5000, log=False, color='r')
    imageHisList = list(imageHisTuple)
    imageHisAverageList[0] += imageHisList[0]
    numberOfH_CT += 1
    
imageHisAverageList[0] /= numberOfH_CT

for i in range(0,5000):
    if(i % 2 == 0):
        imageHisAverageList[0][i] = int(imageHisAverageList[0][i]) + 1
    else:
        imageHisAverageList[0][i] = int(imageHisAverageList[0][i])

# sum = 0
# for i in range(0,2000):
#         sum += imageHisAverageList[0][i]
# print(sum)
print("1")
# 평균히스토그램을 이미지로 만들기
histoAverageImageArr = imageArr
index = 0
count = 0
for i in range (len(imageArr)):
    for j in range (len(imageArr[1])):
        for k in range (len(imageArr[0])):
            if(count > imageHisAverageList[0][index]):
                index += 1
                count = 1
            histoAverageImageArr[i][j][k] = imageHisAverageList[1][index]
            count += 1
            
print("2")
referenceImage = sitk.GetImageFromArray(histoAverageImageArr)
# referenceImage = sitk.Cast(referenceImage, sitk.sitkInt16)
dirUserNames = os.listdir("./Match_source/H/")
for dirUserName in dirUserNames:
     #if 'Quantile' in dirUserName:
      #  continue
    print(dirUserName)
    imageMovingBefore, imageArrMovingBefore, readerMovingBefore = readDicom('./Match_source/H/' + dirUserName)
#     if 'DI_' or 'RP_' in dirUserName:
#         imageMovingBefore = sitk.Cast(imageMovingBefore, sitk.sitkInt16)
    imageMovingBefore, imageArrMovingBefore = thresholdImage(imageArrMovingBefore, 'H')
    imageArrMovingBefore = resize(imageArrMovingBefore.astype(np.float32), (600, 600, 600), anti_aliasing = False)
    imageMovingBefore2 = sitk.GetImageFromArray(imageArrMovingBefore)
    matcher = sitk.HistogramMatchingImageFilter()
    if (image.GetPixelID() in (sitk.sitkUInt8, sitk.sitkInt8)):
        matcher.SetNumberOfHistogramLevels(128)
    else:
        matcher.SetNumberOfHistogramLevels(1024)
    matcher.SetNumberOfMatchPoints(15)
    matcher.ThresholdAtMeanIntensityOn()
    imageMovingAfter = matcher.Execute(imageMovingBefore2, referenceImage)
    imageArrMovingAfter = sitk.GetArrayFromImage(imageMovingAfter)
    #saveDicom(imageArrMovingAfter,'./Match_result/HtoJW/' + dirUserName + '/', imageMovingBefore2+"600", readerMovingBefore)
    result = resize(imageArrMovingAfter.astype(np.float32), (633, 700, 700), anti_aliasing = False)
    saveDicom(result,'./Match_result/HtoJW/' + dirUserName + '/', imageMovingBefore, readerMovingBefore)
    # 9 155 156 돌리기