# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 09:36:34 2020

@author: shawn
"""
import os
import argparse
from sklearn.model_selection import train_test_split

import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2 as cv
from ranger import Ranger
from PIL import Image
from lookahead import Lookahead
from radam import RAdam
from adam import Adam
from testmodel import testmodel
from torchvision.models import resnet18,vgg16,mnasnet0_5,shufflenet_v2_x1_0,SqueezeNet
from mobilenetv3 import mobilenetv3_small,mobilenetv3_large
from efficientnet_pytorch import EfficientNet
from torchvision import transforms

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchsummary import summary



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



def readfile(path):
    if (path==path0):    
        image_dir = os.listdir(path)
        x=np.zeros((len(image_dir), 400, 400, 3),dtype=np.uint8)
        y=np.ones(len(image_dir))
        for i ,file in enumerate(image_dir):
           path=os.path.join(path0, file)
           image = Image.open(path)
           image = image.resize((400, 400))    
           image = np.array(image)
           x[i,:,:]=image
        return x,y
    elif(path==path1):
        image_dir = os.listdir(path)
        x=np.zeros((len(image_dir), 400, 400, 3),dtype=np.uint8)
        y=np.zeros(len(image_dir))
        for i ,file in enumerate(image_dir):
           path=os.path.join(path1, file)
           image = Image.open(path)
           image = image.resize((400, 400))    
           image = np.array(image)
           x[i,:,:]=image
        return x,y
        





"""--------------------前處理-----------------------"""



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
testx=trainx4
testy=trainy4
print(trainx.shape)
print(trainy.shape)
plt.figure(dpi=600)
plt.imshow(trainx[100])
plt.show()




#X_train, X_test, Y_train, Y_test = train_test_split(trainx,trainy,random_state=40, test_size=0.2)



# X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test,random_state=40, test_size=0.5)

size=224#輸入尺寸

train_transform = transforms.Compose([
 
    transforms.ToTensor(), # 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)取值范围为[0, 255]的PIL.Image，转换成形状为[C, H, W]，取值范围是[0, 1.0]的torch.FloadTensor；形状为[H, W, C]的numpy.ndarray，转换成形状为[C, H, W]，取值范围是[0, 1.0]的torch.FloadTensor。
    transforms.Resize((size,size)),
    #transforms.CenterCrop((224,224)),
    # transforms.RandomOrder([
    #     transforms.RandomHorizontalFlip(p=0.5), #依概率p垂直翻转
    #     transforms.RandomVerticalFlip(p=0.5), #随机旋转
    #     transforms.RandomRotation(180),]),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),#做正規化[-1~1]之間
    # transforms.Normalize(mean=[0.5],std=[0.5]),#做正規化[-1~1]之間
    #transforms.RandomErasing( p=0.5 , scale=(0.02 , 0.33) , ratio=(0.3 , 3.3) , value=0 , inplace=False ),#value=0黑色,=(254/255, 0, 0)紅色,='random'  #
    #transforms.ColorJitter(brightness=0.2, contrast=0, saturation=0, hue=0)  #調整亮度效果變差

])



batch_size=64
num_epoch=100

train_set=ImgDataset(trainx, trainy,train_transform)
train_loader=DataLoader(train_set,batch_size=batch_size,shuffle=True,pin_memory=True)
val_set=ImgDataset(testx,testy,train_transform)
val_loader=DataLoader(val_set,batch_size=batch_size,shuffle=False,pin_memory=True)





"""-----------------建置模型------------------"""
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

#model = vgg16(pretrained=False,progress=False).cuda()
#model = shufflenet_v2_x1_0(pretrained=False,progress=False,num_classes=2).to(device)
#model=resnet18(pretrained=False,progress=False).to(device)#num_classes=2
#model=mobilenetv3_small(num_classes=1000,width_mult=1).cuda()
#model = testmodel().to(device)
model = SqueezeNet().to(device)
#model=mobilenet_v3_small(pretrained=True,progress=False).cuda()#使用pretrained model
#model=EfficientNet.from_pretrained('efficientnet-b0').cuda()

#model.load_state_dict(torch.load('./modellast.pth'))#使用之前訓練的model

loss = nn.CrossEntropyLoss()
#optimizer0 = Adam(model.parameters(), lr=1E-3,use_gc=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1E-3,amsgrad=False) # optimizer 使用 Adam
#optimizer0 = RAdam(model.parameters(), lr=1E-3)
#optimizer = Lookahead(optimizer0, k=5, alpha=0.5) # Initialize Lookahead
#optimizer = Ranger(model.parameters(),use_gc=False)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001,momentum=0.002)#,momentum=0.002
#optimizer = torch.optim.AdamW(model.parameters(), lr=0.1) # optimizer 使用 Adamw
#optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5, last_epoch=-1)#等間格調整學習率


#模型數據顯示

summary(model,input_size=(3,size,size))



"""--------------------訓練模型---------------"""
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
    
    model.train()
    for i, data in enumerate(train_loader):
        
        optimizer.zero_grad() # 用 optimizer 將 model 參數的 gradient 歸零
        train_pred = model(data[0].to(device)) # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數
        batch_loss = loss(train_pred, data[1].to(device)) # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上）
        batch_loss.backward() # 利用 back propagation 算出每個參數的 gradient
        optimizer.step() # 以 optimizer 用 gradient 更新參數值
        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()
        
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].to(device))
            batch_loss = loss(val_pred, data[1].to(device))
            
            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()

    train_point.append((train_acc/train_set.__len__(),train_loss/train_set.__len__()))
    test_point.append((val_acc/val_set.__len__(),val_loss/val_set.__len__()))
    
    
    print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f acc Loss: %3.6f val Acc: %3.6f val Loss: %3.6f ' % \
          (epoch + 1, num_epoch, time.time()-epoch_start_time, \
           train_acc/train_set.__len__(), train_loss/train_set.__len__(),\
               val_acc/val_set.__len__(),val_loss/val_set.__len__()))
    
    if(val_acc/val_set.__len__()>best_acc):
        best_acc=val_acc/val_set.__len__()
        print('best_acc:',best_acc)
        torch.save(model.state_dict(), 'ours.pth')
    
    # if((epoch+1)==100):
    #     print("SGD optimizer")
    #     optimizer = torch.optim.SGD(model.parameters(), lr=0.0001,momentum=0.002)
    
    if((epoch+1)==num_epoch):
        print('save_model')
        torch.save(model.state_dict(), 'oursfinal3.pth')
    
trainp = list(zip(*train_point))
testp= list(zip(*test_point))

plt.figure(figsize=(8,8),dpi=600)
plt.subplot(2,2,1, title='Accuracy (train)').plot(trainp[0])
plt.subplot(2,2,2, title='loss (train)').plot(trainp[1])
plt.subplot(2,2,3, title='Accuracy (test)').plot(testp[0])
plt.subplot(2,2,4, title='loss (test)').plot(testp[1])
plt.show()

np.savetxt('ours_train.csv', train_point,delimiter=',',fmt = '% s')
np.savetxt('ours_test.csv', test_point,delimiter=',',fmt = '% s')

















