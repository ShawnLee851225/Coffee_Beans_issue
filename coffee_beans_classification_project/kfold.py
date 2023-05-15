# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 09:42:31 2022

@author: Shawn
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

trainx=np.load('./numpydata/train224x.npy')
trainy=np.load('./numpydata/train224y.npy')
augtrainx=np.load('./numpydata/trainaug224x.npy')
augtrainy=np.load('./numpydata/trainaug224y.npy')
trainx=np.concatenate([trainx,augtrainx])
trainy=np.concatenate([trainy,augtrainy])
# print(trainx.shape)
# print(trainy.shape)
# plt.figure(dpi=600)
# plt.imshow(trainx[100])
# plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(trainx,trainy,random_state=40, test_size=0.2)
print(X_train.shape)

trainx4 = X_test
trainy4 = Y_test
np.save('./numpydata/train224x4',trainx4)
np.save('./numpydata/train224y4',trainy4)
print(trainx4.shape)

trainx0 = X_train[0:3701]
trainy0 = Y_train[0:3701]
np.save('./numpydata/train224x0',trainx0)
np.save('./numpydata/train224y0',trainy0)
print(trainx0.shape)

trainx1 = X_train[3701:7402]
trainy1 = Y_train[3701:7402]
np.save('./numpydata/train224x1',trainx1)
np.save('./numpydata/train224y1',trainy1)
print(trainx1.shape)
  
trainx2 = X_train[7402:11103]
trainy2= Y_train[7402:11103]
np.save('./numpydata/train224x2',trainx2)
np.save('./numpydata/train224y2',trainy2)
print(trainx2.shape) 

trainx3 = X_train[11103:14803]
trainy3= Y_train[11103:14803]
np.save('./numpydata/train224x3',trainx3)
np.save('./numpydata/train224y3',trainy3)
print(trainx3.shape)   