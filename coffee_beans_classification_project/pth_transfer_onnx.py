# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 13:59:15 2022

@author: shawn
"""

import torch
import torch.onnx as onnx

modelpath = './resnet985050.pth'
model = torch.load(modelpath)