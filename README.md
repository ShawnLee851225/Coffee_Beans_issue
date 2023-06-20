# Coffee_Beans_issue
About coffee beans database with Deep learning

## Model parameter setting
image_size =50*50, batch size=64, num classes =2, num epoch=25

optimizer = Adam, loss = CEloss, lr = 1e-3

normalization =>mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]

## MobilenetV3small model training accuracy and loss
<img src=./CNN_classification_project/training_process_data/mobilenetv3_small.png width=80% />

## MobilenetV3small model confusion_matrix
<img src=./CNN_classification_project/training_process_data/mobilenetv3_smallconfusion_matrix.png width=80% />

## BI Visual
<img src=./CNN_classification_project/training_process_data/BItrainingdata.png width=100% />

## Resnet18 model training accuracy and loss
<img src=./CNN_classification_project/training_process_data/resnet18.png width=100% />

## Resnet18 model confusion_matrix
<img src=./CNN_classification_project/training_process_data/resnet18confusion_matrix.png width=80% />