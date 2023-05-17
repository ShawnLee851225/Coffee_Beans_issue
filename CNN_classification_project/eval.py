# -*- coding: utf-8 -*-
"""
Created on 2023/05/10

@author: Shawn YH Lee
"""

"""----------import package----------"""
from main import ds_preprocessing,model_select,device_auto_detect,\
loss_select
import argparse
import numpy as np
import torch
from torchsummary import summary
from tqdm import tqdm

"""----------import package end----------"""

"""----------module switch setting----------"""
tqdm_module = True #progress bar
argparse_module = True  #don't False
Image_transfer_np = False   #transfer dataset to np if not .npy
torchsummary_module = False  #model Visual
check_image_module = False  #Check image is normal
show_line_graph_switch = True 
save_training_progress_csv_switch =True  
"""----------module switch setting end----------"""

"""----------variable init----------"""
#train_point=[]
test_point=[]
#best_acc=0.0
metrics=np.zeros((2,2),dtype=np.int32)
label_map =['bad','good']
"""----------variable init end----------"""

"""----------argparse init----------"""
if argparse_module:    
    parser = argparse.ArgumentParser(description = 'train model')
    parser.add_argument('--database_path0',type=str,default='./database/bad/',help='label 0')
    parser.add_argument('--database_path1',type=str,default='./database/good/',help='label 1')
    parser.add_argument('--modelpath',type=str,default='./model/',help='output model save path')
    parser.add_argument('--numpy_data_path',type=str,default='./numpydata/',help='output numpy data')
    parser.add_argument('--training_data_path',type=str,default='./training_process_data/',help='output training data path')
    parser.add_argument('--image_size',type=int,default= 400,help='image size')
    parser.add_argument('--num_classes',type=int,default= 2,help='num classes')
    parser.add_argument('--batch_size',type=int,default= 64,help='batch_size')
    parser.add_argument('--num_epoch',type=int,default= 1,help='num_epoch')
    parser.add_argument('--model',type= str,default='resnet18',help='model')
    parser.add_argument('--optimizer',type= str,default='Ranger',help='optimizer')
    parser.add_argument('--loss',type= str,default='CrossEntropyLoss',help='Loss')
    parser.add_argument('--lr',type= int,default=1e-3,help='learningrate')

    args = parser.parse_args()
"""----------argparse init end----------"""

"""----------tqdm init----------"""
if tqdm_module:
    pbar = tqdm(range(args.num_epoch),desc='Epoch',unit='epoch',maxinterval=1)
"""----------tqdm init end----------"""

"""----------function----------"""
def model_eval():
    test_acc = 0.0
    test_loss = 0.0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            test_pred = model(data[0].to(device))
            batch_loss = loss(test_pred, data[1].to(device))
            test_acc += np.sum(np.argmax(test_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            test_loss += batch_loss.item() #.item() return 較準確的值
            test_acc_percent = test_acc/test_set.__len__()
            test_loss_average = test_loss/test_set.__len__()
            test_point.append((test_acc_percent,test_loss_average))
            pbar.set_postfix({'Test Acc':test_acc_percent,'Test loss':test_loss_average})   

            """------計算metrics-------"""
            y=data[1].numpy()
            y_pred=np.argmax(test_pred.cpu().data.numpy(), axis=1)
            for j in range(len(y)):
                metrics[y[j]][y_pred[j]]=metrics[y[j]][y_pred[j]]+1  
            """-----------------------"""
def confusion_matrix(metrics):
    TP=metrics[0][0]
    FN=metrics[1][0]
    FP=metrics[0][1]
    TN=metrics[1][1]
    accuracy=(TP+TN)/(TP+TN+FP+FN)
    recall=TP/(TP+FN)
    precision=TP/(TP+FP)
    F1_score=2*precision*recall/(precision+recall)
    print(f'accuracy:{accuracy},precision:{precision},\
          recall:{recall},F1_score:{F1_score}')
def load_model(model):
    model.load_state_dict(torch.load(args.modelpath+args.model+'.pth'))
def create_confusion_matrix_picture():
    import matplotlib.pyplot as plt
    plt.imshow(metrics, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('confusion_matrix-'+args.model)
    plt.colorbar()
    plt.xlabel('Prediction')
    plt.ylabel('Ture')
    tick_marks = np.arange(len(label_map))
    plt.xticks(tick_marks, label_map, rotation=-45)
    plt.yticks(tick_marks, label_map)
    iters = np.reshape([[[i,j] for j in range(len(label_map))] for i in range(len(label_map))],(metrics.size,2))
    for i, j in iters:
        plt.text(j, i, format(metrics[i, j]),va='center',ha='center')
    plt.tight_layout()
    plt.savefig(args.training_data_path+args.model+'confusion_matrix'+'.png')
    #plt.show()
"""----------function end----------"""

if __name__ == '__main__':
    test_set,test_loader =ds_preprocessing()
    device = device_auto_detect()
    model = model_select(device)
    load_model(model)
    loss = loss_select()

    if torchsummary_module:
        summary(model,input_size=(3,args.image_size,args.image_size))

    if tqdm_module:
        for epoch in pbar:
            model_eval()
    else:
        for epoch in range(args.num_epoch):
            model_eval()
    create_confusion_matrix_picture()
    
    
