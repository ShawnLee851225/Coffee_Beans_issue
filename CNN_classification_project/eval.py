# -*- coding: utf-8 -*-
"""
Created on 2023/05/10

@author: Shawn YH Lee
"""

"""----------import package----------"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import os
from PIL import Image
from torchsummary import summary
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
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
    parser.add_argument('--image_size',type=int,default= 50,help='image size')
    parser.add_argument('--num_classes',type=int,default= 2,help='num classes')
    parser.add_argument('--batch_size',type=int,default= 64,help='batch_size')
    parser.add_argument('--num_epoch',type=int,default= 1,help='num_epoch')
    parser.add_argument('--model',type= str,default='mobilenetv3_small',help='mobilenetv3_small, resnet18')
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
train_transform = transforms.Compose([
 
    transforms.ToTensor(), # 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)取值范围为[0, 255]的PIL.Image，转换成形状为[C, H, W]，取值范围是[0, 1.0]的torch.FloadTensor；形状为[H, W, C]的numpy.ndarray，转换成形状为[C, H, W]，取值范围是[0, 1.0]的torch.FloadTensor。
    transforms.Resize((args.image_size,args.image_size)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),#做正規化[-1~1]之間
])
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
    if (path==args.database_path0):  
        image_dir = os.listdir(path)
        x=np.zeros((len(image_dir), args.image_size, args.image_size, 3),dtype=np.uint8)
        n= len(image_dir)
        y=np.zeros(len(image_dir))

        for i ,file in enumerate(image_dir):
           path=os.path.join(args.database_path0, file)
           image = Image.open(path)
           image = image.resize((args.image_size, args.image_size))    
           image = np.array(image)
           x[i,:,:]=image
        return x,y
    elif(path==args.database_path1):
        image_dir = os.listdir(path)
        x=np.zeros((len(image_dir), args.image_size, args.image_size, 3),dtype=np.uint8)
        n= len(image_dir)
        y=np.ones(len(image_dir))
        for i ,file in enumerate(image_dir):
           path=os.path.join(args.database_path1, file)
           image = Image.open(path)
           image = image.resize((args.image_size, args.image_size))    
           image = np.array(image)
           x[i,:,:]=image
        return x,y
def ds_preprocessing():
    print("Read Image from root")
    bad_x,bad_y = readfile(args.database_path0)
    good_x,good_y = readfile(args.database_path1)
    """----------Image_transfer_np----------"""
    if Image_transfer_np:
        print("Image_transfer_np")
        np.save(args.numpy_data_path+'bad_x.npy',bad_x)
        np.save(args.numpy_data_path+'bad_y.npy',bad_y)
        np.save(args.numpy_data_path+'good_x.npy',good_x)
        np.save(args.numpy_data_path+'good_y.npy',good_y)
    
        bad_x=np.load(args.numpy_data_path+'bad_x.npy')
        bad_y=np.load(args.numpy_data_path+'bad_y.npy')
        good_x=np.load(args.numpy_data_path+'good_x.npy')
        good_y=np.load(args.numpy_data_path+'good_y.npy')
    """----------Image_transfer_np end----------"""
    trainx=np.concatenate([bad_x,good_x])
    trainy=np.concatenate([bad_y,good_y])
    train_set=ImgDataset(trainx, trainy,train_transform)
    train_loader=DataLoader(train_set,batch_size = args.batch_size,shuffle=True,pin_memory=True)
    if check_image_module:
        #check image
        import matplotlib.pyplot as plt
        plt.figure(dpi=600)
        #close 座標刻度與座標軸
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.imshow(trainx[0])
        plt.show()
    return train_set,train_loader
def device_auto_detect():
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    print(f'device:{device}')
    return device
def model_select(device='cpu'):
    if args.model == 'resnet18':
        from torchvision.models import resnet18
        model = resnet18(progress=False,num_classes=args.num_classes).to(device)
    elif args.model == 'mobilenetv3_small':
        from torchvision.models import mobilenet_v3_small
        model = mobilenet_v3_small(progress=False,num_classes= args.num_classes).to(device)
    else:
        print("dont know the model name")
    print(f"Select model:{args.model}")
    return model
def loss_select():
    if args.loss == 'CrossEntropyLoss':
        loss = nn.CrossEntropyLoss()
    else:
        print("unknown loss function")
    print(f"Select loss function:{args.loss}")
    return loss
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
    test_set,test_loader = ds_preprocessing()
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
    
    
