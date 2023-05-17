# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 16:02:45 2021

@author: shawn
"""

import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

"""--------------讀取檔案--------------"""

#good資料夾
path='./datarotate/good'#檔案位置
size=224
trainx=np.zeros((len(os.listdir(path)),size,size, 3),dtype=np.uint8)
for i,file in enumerate(os.listdir(path)):
    print(file)
    image=Image.open(os.path.join(path, file))#讀取影像
    image=image.resize((size,size),0)
    image=np.array(image)#轉numpy矩陣
    trainx[i,:,:]=image

np.save('good224x.npy',trainx)
print(trainx.shape)
trainy=np.array([0]*len(trainx))
np.save('good224y.npy',trainy)
print(len(trainy))


#bad資料夾
path='./datarotate/bad'#檔案位置
size=224
trainx=np.zeros((len(os.listdir(path)),size,size, 3),dtype=np.uint8)
for i,file in enumerate(os.listdir(path)):
    print(file)
    image=Image.open(os.path.join(path, file))#讀取影像
    image=image.resize((size,size),0)
    image=np.array(image)#轉numpy矩陣
    trainx[i,:,:]=image

np.save('bad224x.npy',trainx)
print(trainx.shape)
trainy=np.array([1]*len(trainx))
np.save('bad224y.npy',trainy)
print(len(trainy))


"""----------------------整合------------"""


trainx=np.concatenate([np.load('good224x.npy'),np.load('bad224x.npy')])
trainy=np.concatenate([np.load('good224y.npy'),np.load('bad224y.npy')])
np.save('trainrotate224x', trainx)
np.save('trainrotate224y',trainy)




