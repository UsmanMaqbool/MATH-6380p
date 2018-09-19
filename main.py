import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms,utils
import pickle
import matplotlib.pyplot as plt
import time
import os
import copy

########## loading data ##############

data_transform=transforms.Compose([transforms.Resize((256,256)),
                                   transforms.RandomCrop((224,224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])
                                   ])

raphael_dataset=datasets.ImageFolder(root='./data/Raphael-jpg/',
                                          transform=data_transform)
#raphael_dataset=datasets.ImageFolder(root='./data/test/',
#                                          transform=data_transform)

data_loader = torch.utils.data.DataLoader(raphael_dataset,
                                             batch_size=1, shuffle=True)

#print(raphael_dataset.imgs)

imgcount=len(raphael_dataset)
#print(imgcount)

########## generate feature ############

def generate_feature(img, net,num_ftrs):

    out= net(img)
    feature = out
    
    return feature


pretrained_model = torchvision.models.vgg16(pretrained=True)
print(pretrained_model)

for param in pretrained_model.parameters():
    param.requires_grad = False
#num_ftrs = pretrained_model.classifier._modules[3].Linear.out_features
num_ftrs = 4096


pretrained_model.classifier = nn.Sequential(*list(pretrained_model.classifier.children())[:-3]) # to relu5_3
print('modified')
print(pretrained_model)

featureMat=np.empty((imgcount,num_ftrs))
labelVec = np.zeros(imgcount)

print('generating features...')

for i,(img,label) in enumerate(data_loader):

    featureMat[i,:]=generate_feature(img,pretrained_model,num_ftrs)
    labelVec[i]=int(label)

f = open('features_train.pkl','wb')
pickle.dump(featureMat, f)
f.close()

f = open('label_train.pkl','wb')
pickle.dump(labelVec, f)
f.close()
#pretrained_model.fc = nn.Linear(num_ftrs, 2)


#model_conv = model_conv.to(device)

#criterion = nn.CrossEntropyLoss()

#optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)





