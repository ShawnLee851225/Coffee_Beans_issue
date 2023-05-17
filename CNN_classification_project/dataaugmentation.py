# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 06:36:58 2021

@author: shawn
"""

import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageOps
from torchvision import transforms
# #good資料夾
# path='./good'#檔案位置
# size=400
# trainx=np.zeros((len(os.listdir(path)*3),size,size, 3),dtype=np.uint8)
# trainy=np.array([0]*len(os.listdir(path))*3,dtype=np.uint8)
# for i,file in enumerate(os.listdir(path)):
#     print(file)
#     image=Image.open(os.path.join(path, file))#讀取影像
#     image=image.resize((size,size),0)
#     image1=image.transpose(Image.FLIP_LEFT_RIGHT)#水平翻轉
#     image2=image.transpose(Image.FLIP_TOP_BOTTOM)#垂直翻轉
#     image3=image.transpose(Image.ROTATE_180)#逆旋轉180翻轉
#     image1=np.array(image1)#轉numpy矩陣
#     image2=np.array(image2)#轉numpy矩陣
#     image3=np.array(image3)#轉numpy矩陣
#     trainx[i*3,:,:]=image1
#     trainx[i*3+1,:,:]=image2
#     trainx[i*3+2,:,:]=image3

# #bad資料夾
# path='./bad'#檔案位置
# size=400
# testx=np.zeros((len(os.listdir(path)*3),size,size, 3),dtype=np.uint8)
# testy=np.array([1]*len(os.listdir(path))*3,dtype=np.uint8)
# for i,file in enumerate(os.listdir(path)):
#     print(file)
#     image=Image.open(os.path.join(path, file))#讀取影像
#     image=image.resize((size,size),0)
#     image1=image.transpose(Image.FLIP_LEFT_RIGHT)#水平翻轉
#     image2=image.transpose(Image.FLIP_TOP_BOTTOM)#垂直翻轉
#     image3=image.transpose(Image.ROTATE_180)#逆旋轉180翻轉
#     image1=np.array(image1)#轉numpy矩陣
#     image2=np.array(image2)#轉numpy矩陣
#     image3=np.array(image3)#轉numpy矩陣
#     testx[i*3,:,:]=image1
#     testx[i*3+1,:,:]=image2
#     testx[i*3+2,:,:]=image3
    
# np.save('data_augx', np.concatenate([trainx, testx]))
# np.save('data_augy', np.concatenate([trainy, testy]))

""""----一次完成"""
path='./database'
for i,filename in enumerate(os.listdir(path)):
    filepath = os.path.join(path, filename)
    for j,imagename in enumerate(os.listdir(filepath)):
        print(imagename)
        imagepath = os.path.join(filepath,imagename)
        
        # image=Image.open(imagepath)#讀取影像
        # image1 = transforms.functional.rotate(image, 40)
        # image1.save('./dataaug/'+filename+'/rotate40'+imagename)
        """------------opencv去除背景-------------"""
        image = cv.imread(imagepath)
        h,w = image.shape[:2]
        center = (w/2,h/2)
        M = cv.getRotationMatrix2D(center, 320, 1)
        image1 = cv.warpAffine(image, M, (w,h),borderValue = (255,255,255))
        cv.imwrite('./datarotate/'+filename+'/rotate320'+imagename, image1)
        
        # image1=image.transpose(Image.FLIP_LEFT_RIGHT)#水平翻轉
        # image2=image.transpose(Image.FLIP_TOP_BOTTOM)#垂直翻轉
        # image3=image.transpose(Image.ROTATE_180)#逆旋轉180翻轉
        # image1.save('./dataaug/'+filename+'/Vertical'+imagename)
        # image2.save('./dataaug/'+filename+'/Horizontal'+imagename)
        # image3.save('./dataaug/'+filename+'/Rotate'+imagename)
        
        
        
        
