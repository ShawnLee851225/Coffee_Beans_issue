# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 23:56:49 2021

@author: shawn
"""


import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image

path0 = 'D:\\training-data-jpg\\good\\Set04-good.01.01.jpg' #絕對路徑
path1 = './good/Set04-good.01.01.jpg'  #相對路徑



image = cv.imread(path1,cv.IMREAD_COLOR)#讀取圖片
print(image.shape)#印出圖片大小
image = image[:,:,::-1]
plt.imshow(image)
plt.show()


# cv.imshow('image',image)#顯示圖片
# cv.waitKey(0)   #等待指令
# cv.destroyAllWindows()   #關閉視窗

# image = Image.open(path1)
# plt.imshow(image)
# plt.show()


np.save("trainx.npy",trainx)
np.save("trainy.npy",trainy)

trainx=np.load('trainx.npy')
trainy=np.load('trainy.npy')





