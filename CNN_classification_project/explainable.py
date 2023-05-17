# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 11:59:14 2022

@author: Shawn
"""

import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np
import os, json
import torch
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F
from lime import lime_image
from testmodel import testmodel
from sklearn.model_selection import train_test_split
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from efficientnet_pytorch import EfficientNet
from torchvision.models import mobilenet_v3_small,resnet18






size = 224

#model = EfficientNet.from_pretrained('efficientnet-b0')
#model = mobilenet_v3_small()
#model = resnet18()
model = testmodel()
#model.load_state_dict(torch.load('./model.pth'))

model.load_state_dict(torch.load('./ourmodel/newmodelfinal.pth'))

path = "./database/bad/Set05-bad.22.enhanced.08.jpg"

image = Image.open(path)






def get_pil_transform(): 
    transf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((size, size)),
        transforms.ToPILImage()

    ])    

    return transf

def get_preprocess_transform():
        
    transf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((size,size)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        # transforms.Normalize(mean=[0.5],std=[0.5]),
    ])    

    return transf    

pill_transf = get_pil_transform()
preprocess_transform = get_preprocess_transform()  

def batch_predict(images):
    model.eval()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device)
    
    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()
def segmentation(input):
    # 利用 skimage 提供的 segmentation 將圖片分成 100 塊                                                                                                                                      
    return slic(input, n_segments=100, compactness=1, sigma=1)   

test_pred = batch_predict([pill_transf(image)])
test_pred.squeeze().argmax()
print(test_pred[0,0],test_pred[0,1])                                                                                                                        
print(test_pred.squeeze().argmax())


explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(np.array(pill_transf(image)), 
                                         batch_predict, # classification function
                                         top_labels=2, 
                                         hide_color=0, 
                                         num_samples=25000,segmentation_fn=segmentation) 
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=15, hide_rest=True)
img_boundry1 = mark_boundaries(temp/255, mask)


temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=15, hide_rest=False)
img_boundry2 = mark_boundaries(temp/255, mask)


plt.figure(dpi=600)
plt.imshow(image)
plt.xticks([])
plt.yticks([])
plt.show()
plt.imshow(img_boundry2)
plt.xticks([])
plt.yticks([])
plt.show()
plt.imshow(img_boundry1)
plt.xticks([])
plt.yticks([])
plt.show()
# plt.subplot(1,3,1)
# plt.imshow(image)
# plt.xticks([])
# plt.yticks([])
# plt.subplot(1,3,2)
# plt.imshow(img_boundry2)
# plt.xticks([])
# plt.yticks([])
# plt.subplot(1,3,3)
# plt.imshow(img_boundry1)
# plt.xticks([])
# plt.yticks([])

# plt.show()































