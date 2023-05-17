# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 13:18:22 2021

@author: shawn
"""

import numpy as np

goodx=np.load('good128x.npy')
goody=np.load('goody.npy')
badx=np.load('bad128x.npy')
bady=np.load('bady.npy')

trainx=np.concatenate([goodx,badx])
trainy=np.concatenate([goody,bady])


np.save('train128x',trainx)
np.save('trainy',trainy)
