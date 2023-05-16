# -*- coding: utf-8 -*-
"""
Created on 2023/05/16

@author: Shawn YH Lee
"""
"""----------import package----------"""
import os
import numpy as np
import argparse
from PIL import Image
from sklearn import svm
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
"""----------import package end----------"""

"""----------module switch setting----------"""
argparse_module = True  #don't False
tqdm_module =True
torchsummary_module = False
"""----------module switch setting end----------"""

"""----------argparse init----------"""
if argparse_module:    
    parser = argparse.ArgumentParser(description = 'train model')
    parser.add_argument('--database_path',type=str,default='../coffee_beans_classification_project/database/')
    parser.add_argument('--modelpath',type=str,default='./model/',help='output model save path')
    parser.add_argument('--training_data_path',type=str,default='./training_process_data/',help='output training data path')
    parser.add_argument('--image_size',type=int,default= 30,help='image size')
    parser.add_argument('--num_classes',type=int,default= 2,help='num classes')

    args = parser.parse_args()
"""----------argparse init end----------"""
"""----------function----------"""
def data_process(classes_dir):
    x = np.zeros(0)
    y = np.zeros(0,dtype=np.uint8)
    for i, classes in enumerate(classes_dir):
        class_dir = os.path.join(args.database_path,classes)
        file_path = os.listdir(class_dir)
        length = len(file_path)
        # process image
        image_array = np.zeros((length,args.image_size,args.image_size,3),dtype=np.uint8)
        # process label    
        y_len = np.full(length,i,dtype=np.uint8)
        y = np.concatenate((y,y_len),axis= 0)
 
        for j,file in enumerate(file_path):
            path = os.path.join(class_dir,file)
            image = Image.open(path)
            image = image.resize((args.image_size, args.image_size))
            image = np.array(image)
            image_array[i,:,:]=image
        if i ==0 :
            x = image_array
        else:
            x =np.concatenate((x,image_array),axis= 0)
    X_train,X_test,y_train,y_test =train_test_split(x,y,test_size=0.2,random_state= 10,shuffle=True)
    return X_train,X_test,y_train,y_test 

"""----------function end----------"""

"""----------main----------"""
if __name__ == '__main__':
    X_train,X_test,y_train,y_test = data_process(os.listdir(args.database_path))
    
    model = svm.SVC(kernel='linear', C= 1.0)
    #影像降維
    X_train=X_train.reshape(len(X_train),-1)
    X_test = X_test.reshape(len(X_test),-1)
    print(X_train.dtype,y_train.dtype)
    print('model fit')
    model.fit(X_train,y_train)
    #pred = model.predict(X_test)
    acc = model.score(X_test,y_test)
    print(f'accuracy: {acc}')


        


