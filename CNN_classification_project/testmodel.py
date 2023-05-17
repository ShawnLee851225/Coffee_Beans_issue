# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 23:17:57 2021

@author: shawn
"""
import torch
import torch.nn as nn
import math
"""------------h-swish函數modual實現-------------"""
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
"""------------SElayer-------------"""
class SELayer(nn.Module):
    def __init__(self, channel,hiddenlayer):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel,hiddenlayer),
                nn.ReLU(inplace=True),
                nn.Linear(hiddenlayer, channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
"""---------------CNN函數--------------------"""
def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup,3,stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish(),
    )
def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
    )

"""------------------------------------------------------------------------"""


class testmodel(nn.Module):
    def __init__(self, num_classes=1000):
        super(testmodel, self).__init__()
        self.cnn1=conv_3x3_bn(inp=3, oup=16, stride=2)#pw
        self.skipblock = nn.Sequential(
            nn.AdaptiveAvgPool2d((56,56)),
            nn.BatchNorm2d(16),
            h_swish(),
            nn.Conv2d(16, 48, 3,2,1, groups=16, bias=False),
            nn.BatchNorm2d(48),
            )
        
        self.cnn2=nn.Sequential(
            # pw
            nn.Conv2d(16, 64, 1, 1, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            #dw
            nn.Conv2d(64,64,3,2,1, groups=64, bias=False),
            nn.BatchNorm2d(64),
            SELayer(channel = 64, hiddenlayer = 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 1,1,0,bias=False),
            nn.BatchNorm2d(32),
            )
        self.cnn3=nn.Sequential(
            #pw
            nn.Conv2d(32, 128,1,1,0,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            #dw
            nn.Conv2d(128,128,3,2,1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            SELayer(channel = 128, hiddenlayer = 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,48, 1,1,0,bias=False),
            nn.BatchNorm2d(48),
            )
        self.cnn4=nn.Sequential(
            #dw
            nn.Conv2d(48,128,3,1,1, bias=False),
            nn.BatchNorm2d(128),
            SELayer(channel = 128, hiddenlayer = 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,48, 1,1,0,bias=False),
            nn.BatchNorm2d(48),
            )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(48, 1024),
            h_swish(),
            nn.Dropout(0.2),#原始0.2
            nn.Linear(1024, 2),
        )
        
    def forward(self, x):
        x1 = self.cnn1(x)
        x2 = self.skipblock(x1)
        x1 = self.cnn2(x1)
        x1 = self.cnn3(x1)
        x1 = x1+x2
        x1 = self.cnn4(x1)+x1
        x1 = self.avgpool(x1)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.classifier(x1)

  
        return x1