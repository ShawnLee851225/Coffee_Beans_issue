# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 04:40:46 2022

@author: shawn
"""
import torch

from tensorboardX import SummaryWriter
from mobilenetv3 import mobilenetv3_small

model = mobilenetv3_small()

dummy_input = torch.rand(1,3,400,400)

with SummaryWriter(comment='outnet') as w:
    w.add_graph(model, (dummy_input, ))