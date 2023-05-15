# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 15:04:53 2022

@author: shawn
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image

from lookahead import Lookahead
from torchvision.models import resnet18,resnet34,resnet50,shufflenet_v2_x1_0
from torchvision import transforms
from efficientnetv2 import effnetv2_s
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchsummary import summary
from efficientnet_pytorch import EfficientNet
from mobilenetv3 import mobilenetv3_small,mobilenetv3_large
from testmodel import testmodel
path0='./training-data-jpg/good/'
path1='./training-data-jpg/bad'

class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X
trainx0=np.load('./numpydata/train224x0.npy')
trainy0=np.load('./numpydata/train224y0.npy')
trainx1=np.load('./numpydata/train224x1.npy')
trainy1=np.load('./numpydata/train224y1.npy')
trainx2=np.load('./numpydata/train224x2.npy')
trainy2=np.load('./numpydata/train224y2.npy')
trainx3=np.load('./numpydata/train224x3.npy')
trainy3=np.load('./numpydata/train224y3.npy')
trainx4=np.load('./numpydata/train224x4.npy')
trainy4=np.load('./numpydata/train224y4.npy')
trainx=np.concatenate([trainx0,trainx1,trainx2,trainx3])
trainy=np.concatenate([trainy0,trainy1,trainy2,trainy3])
testx=trainx0
testy=trainy0
print(testx.shape)
print(testy.shape)

#X_train, X_test, Y_train, Y_test = train_test_split(trainx,trainy,random_state=40, test_size=0.2)
# X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test,random_state=40, test_size=0.5)
size=224

train_transform = transforms.Compose([
 
    transforms.ToTensor(), # 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)取值范围为[0, 255]的PIL.Image，转换成形状为[C, H, W]，取值范围是[0, 1.0]的torch.FloadTensor；形状为[H, W, C]的numpy.ndarray，转换成形状为[C, H, W]，取值范围是[0, 1.0]的torch.FloadTensor。
    transforms.Resize((size,size)),
    transforms.Normalize(mean =[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),#做正規化[-1~1]之間
    # transforms.Normalize(mean=[0.5],std=[0.5]),#做正規化[-1~1]之間
    #transforms.RandomHorizontalFlip(p=0.5), #依概率p垂直翻转
    #transforms.RandomVerticalFlip(p=0.5), #随机旋转
    #transforms.RandomRotation(180),
    #transforms.RandomErasing( p=0.5 , scale=(0.02 , 0.33) , ratio=(0.3 , 3.3) , value='random' , inplace=False ),#value=0黑色,=(254/255, 0, 0)紅色,='random'  #
    #transforms.ColorJitter(brightness=0.2, contrast=0, saturation=0, hue=0)  #調整亮度效果變差

])




batch_size=32
num_epoch=1

train_set=ImgDataset(trainx, trainy,train_transform)
train_loader=DataLoader(train_set,batch_size=batch_size,shuffle=True,pin_memory=True)
val_set=ImgDataset(testx,testy,train_transform)
val_loader=DataLoader(val_set,batch_size=batch_size,shuffle=False,pin_memory=True)
device = torch.device('cpu' if torch.cuda.is_available else 'cpu')
#model = shufflenet_v2_x1_0().to(device)
#model=resnet18(pretrained=False,progress=False,num_classes=1000).to(device)
#model=mobilenet_v3_small(pretrained=False,progress=False)#使用pretrained model
#model=EfficientNet.from_name('efficientnet-b0')
#model = mobilenetv3_small(num_classes=1000,width_mult=1).to(device)
#model = effnetv2_s().to(device)
model = testmodel().to(device)
#model = resnet50().to(device)
#model = mobilenetv3_large()
model.load_state_dict(torch.load('./model0.pth'))#使用之前訓練的model




loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1E-4,amsgrad=True) # optimizer 使用 Adam
#lookahead = Lookahead(optimizer, k=5, alpha=0.5) # Initialize Lookahead
#optimizer = torch.optim.SGD(model.parameters(), lr=0.0001,momentum=0.002)
#optimizer = torch.optim.AdamW(model.parameters(), lr=0.1) # optimizer 使用 Adamw
#torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1, last_epoch=-1)#等間格調整學習率



#模型數據顯示
#summary(model,input_size=(3,size,size)).to(device)



train_point=[]
test_point=[]
metrics=np.zeros((2,2),dtype=np.int32)

best_acc=0.0
count=0
for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0
    
    model.eval()
    """量化步驟"""
    model.fuse_model()
    model.qconfig =torch.ao.quantization.defalt_qconfig

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].to(device))
            batch_loss = loss(val_pred, data[1].to(device))
            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            """------計算metrics-------"""
            y=data[1].numpy()
            y_pred=np.argmax(val_pred.cpu().data.numpy(), axis=1)
            for j in range(len(y)):
                metrics[y[j]][y_pred[j]]=metrics[y[j]][y_pred[j]]+1  
            """-----------------------"""
                          
            val_loss += batch_loss.item()
    train_point.append((train_acc/train_set.__len__(),train_loss/train_set.__len__()))
    test_point.append((val_acc/val_set.__len__(),val_loss/val_set.__len__()))
    """
    for i in range(len(data[0])):
        y=np.argmax(data[i],axis=0)
        y_p=np.argmax(val_pred[i],axis=0)
        metrics[y][y_p]=metrics[y][y_p]+1
    """
    print('[%03d/%03d] %2.2f sec(s)  val Acc: %3.6f val Loss: %3.6f ' % \
          (epoch + 1, num_epoch, time.time()-epoch_start_time, \
               val_acc/val_set.__len__(),val_loss/val_set.__len__()))
print(metrics)
TP=metrics[0][0]
FN=metrics[1][0]
FP=metrics[0][1]
TN=metrics[1][1]
accuracy=(TP+TN)/(TP+TN+FP+FN)
recall=TP/(TP+FN)
precision=TP/(TP+FP)
F1_score=2*precision*recall/(precision+recall)
print('accuracy:',accuracy,'precision:',precision,'recall:',recall,'F1_score:',F1_score)
