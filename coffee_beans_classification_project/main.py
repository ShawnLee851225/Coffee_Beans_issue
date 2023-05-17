# -*- coding: utf-8 -*-
"""
Created on 2023/05/08

@author: Shawn YH Lee
"""
"""----------import package----------"""
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from torchvision import transforms
from tqdm import tqdm

"""----------import package end----------"""

"""----------variable init----------"""
train_point=[]
#test_point=[] not use
best_acc=0.0
#metrics=np.zeros((2,2),dtype=np.int32)
label_map ={
    0:'bad',
    1:'good',
}
"""----------variable init end----------"""

"""----------module switch setting----------"""
tqdm_module = True #progress bar
argparse_module = True  #don't False
Image_transfer_np = False   #transfer dataset to np if not .npy
torchsummary_module = True  #model Visual
check_image_module = False  #Check image is normal
show_line_graph_switch = True 
save_training_progress_csv_switch =True  
"""----------module switch setting end----------"""

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
    parser.add_argument('--num_epoch',type=int,default= 100,help='num_epoch')
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
train_transform = transforms.Compose([
 
    transforms.ToTensor(), # 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)取值范围为[0, 255]的PIL.Image，转换成形状为[C, H, W]，取值范围是[0, 1.0]的torch.FloadTensor；形状为[H, W, C]的numpy.ndarray，转换成形状为[C, H, W]，取值范围是[0, 1.0]的torch.FloadTensor。
    transforms.Resize((args.image_size,args.image_size)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),#做正規化[-1~1]之間
])
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
def optimizer_select(model):
    if args.optimizer =='Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,amsgrad=False)
    elif args.optimizer == 'RAdam':
        from radam import RAdam
        optimizer = RAdam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'Ranger':
        from ranger import Ranger
        optimizer = Ranger(model.parameters(),use_gc=False,lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,momentum=0.002)
    else:
        print("unknown optimizer function")
    print(f"Select optimizer function:{args.optimizer}")
    return optimizer
def loss_select():
    if args.loss == 'CrossEntropyLoss':
        loss = nn.CrossEntropyLoss()
    else:
        print("unknown loss function")
    print(f"Select loss function:{args.loss}")
    return loss
def ds_preprocessing():
    bad_x=np.load(args.numpy_data_path+'bad_x.npy')
    bad_y=np.load(args.numpy_data_path+'bad_y.npy')
    good_x=np.load(args.numpy_data_path+'good_x.npy')
    good_y=np.load(args.numpy_data_path+'good_y.npy')
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
def model_train():
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    model.train()
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()   # 用 optimizer 將 model 參數的 gradient 歸零
        train_pred = model(data[0].to(device))  # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數
        batch_loss = loss(train_pred, data[1].to(device))   # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上）
        batch_loss.backward()   # 利用 back propagation 算出每個參數的 gradient
        optimizer.step()    # 以 optimizer 用 gradient 更新參數值
        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()
    train_acc_percent = train_acc/train_set.__len__()
    train_loss_average = train_loss/train_set.__len__()
    train_point.append((train_acc_percent,train_loss_average))
    pbar.set_postfix({'Train Acc':train_acc_percent,'Train loss':train_loss_average})   
    if (epoch+1) == args.num_epoch:
        save_model()
def device_auto_detect():
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    print(f'device:{device}')
    return device
def save_model():
    torch.save(model.state_dict(), args.modelpath +args.model +'.pth')
def show_line_graph():
    import matplotlib.pyplot as plt

    trainp = list(zip(*train_point))

    plt.figure(figsize=(8,4),dpi=100)
    plt.subplot(2,2,1, title='Accuracy (train)').plot(trainp[0])
    plt.subplot(2,2,2, title='loss (train)').plot(trainp[1])
    plt.savefig(args.training_data_path+args.model+'.png')
    #plt.show() can close
def save_training_progress_csv():
    np.savetxt( args.training_data_path + args.model +'.csv', train_point,delimiter=',',fmt = '% s')
    #np.savetxt('ours_test.csv', test_point,delimiter=',',fmt = '% s') not use
"""----------function end----------"""

"""----------Image_transfer_np----------"""
if Image_transfer_np:
    print("Image_transfer_np")
    bad_x,bad_y = readfile(args.database_path0)
    good_x,good_y = readfile(args.database_path1)
    np.save(args.numpy_data_path+'bad_x.npy',bad_x)
    np.save(args.numpy_data_path+'bad_y.npy',bad_y)
    np.save(args.numpy_data_path+'good_x.npy',good_x)
    np.save(args.numpy_data_path+'good_y.npy',good_y)
"""----------Image_transfer_np end----------"""

"""----------main----------"""
if __name__ == '__main__':
    """----------data preprocessing----------"""
    train_set,train_loader =ds_preprocessing()
    """----------data preprocessing end----------"""

    device = device_auto_detect()
    model = model_select(device=device)
    optimizer = optimizer_select(model)
    loss = loss_select()

    if torchsummary_module:
        summary(model,input_size=(3,args.image_size,args.image_size))
    if tqdm_module:
        for epoch in pbar:
            model_train()
    else:
        for epoch in range(args.num_epoch):
            model_train()
    if show_line_graph_switch:
        show_line_graph()
    if save_training_progress_csv_switch:
        save_training_progress_csv()
        
"""----------main end----------"""