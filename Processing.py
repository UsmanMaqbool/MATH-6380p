
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms,utils
import pickle

img_root='data/Raphael Project final copy/TrainData'
data_List='data/Raphael Project final copy/label.txt'

imgCount = 0

def read_img(path,img_name):
    img = dict()
    img_=Image.open(path+"/"+img_name)
    img_np=np.array(img_)
    img_3=img_np[:,:,:3]
    img[img_name]=img_3
    return img

def generate_DataSet(data_List, img_root):
    imgCount = 0
    labelDict = dict()
    imgDict = dict()
    with open(data_List) as filer:
        for line in filer.readlines():
            splits = line.strip().split()
            img_name=splits[0]
            label = int(splits[1])
            if label not in labelDict:
                labelDict[label] = list()
            labelDict[label].append(img_name)
            imgCount += 1
    print('Total images:', imgCount)

    for i in labelDict.keys():
        label = int(i)
        # print(label)
        imgList = labelDict[i]
        for imgPath in imgList:
            img = read_img(img_root, imgPath)
            if label not in imgDict:
                imgDict[label] = list()
            imgDict[label].append(img)
    return imgDict

def crop_and_resize(im,size):
    im_np = np.array( im )
    H = im_np.shape[0]
    W = im_np.shape[1]
    num_H = H//size
    num_W = W//size
    skip_H = (H-num_H*size)//2
    skip_W = (W-num_W*size)//2
    im_np_2 = im_np[skip_H:skip_H+num_H*size,skip_W:skip_W+num_W*size,:].reshape(num_H,size,num_W,size,im_np.shape[-1]).transpose((0,2,1,3,4)).reshape(-1,size,size,im_np.shape[-1])
    im_np_1 = im_np[0:num_H*size,0:num_W*size,:].reshape(num_H,size,num_W,size,im_np.shape[-1]).transpose((0,2,1,3,4)).reshape(-1,size,size,im_np.shape[-1])
    im_np = np.stack( (im_np_1,im_np_2)).reshape(-1,size,size,im_np.shape[-1])
    #print( im_np.shape )
    return im_np

def resize_img(im,ratio):
    H = im.size[0]
    W = im.size[1]
    return im.resize(( int(H*ratio),int(W*ratio)),resample = Image.LANCZOS )

def get_dataset_cropped(raw_img,size):
    dataset = dict()
    for cls in raw_img:
        dataset[cls] = dict()
        for im in raw_img[cls]:
            for name in im.keys():
                dataset[cls][name] = crop_and_resize(im[name], size)
    return dataset

def get_data_ready(dataset):
    classes = list(dataset.keys())
    i = 0
    for cls in classes:
        img_names = list(dataset[cls].keys())
        for img in img_names:
            if i == 0:
                chunk = dataset[cls][img]
                group = np.repeat(int(img.split(".")[0]),chunk.shape[0])
                label = np.repeat(cls,chunk.shape[0])
            else:
                num_chunk = dataset[cls][img].shape[0]
                chunk = np.concatenate( (chunk,dataset[cls][img]),axis = 0 )
                group = np.concatenate( (group,np.repeat(int(img.split(".")[0]),num_chunk) ),axis = 0 )
                label = np.concatenate( (label,np.repeat(cls,num_chunk) ),axis = 0 )
            i+=1

    return chunk,label,group
def generate_feature(X):
    pretrained_model = torchvision.models.vgg16(pretrained=True)
    # print(pretrained_model)

    for param in pretrained_model.parameters():
        param.requires_grad = False
    # num_ftrs = pretrained_model.classifier._modules[3].Linear.out_features
    num_ftrs = 4096

    pretrained_model.classifier = nn.Sequential(*list(pretrained_model.classifier.children())[:-3])  # to relu5_3
    pretrained_model.eval()

    X_t = torch.from_numpy(X[:].transpose(0, 3, 1, 2)).type(torch.FloatTensor)
    print(type(X_t))


    num = X_t.shape[0] // 100
    X_split = np.array_split(X_t, num)

    for i, xs in enumerate(X_split):
        print("Split " + str(i) + "/" + str(num))
        if i == 0:
            feat = np.squeeze(pretrained_model(xs).numpy())
        else:
            feat = np.concatenate((feat, np.squeeze(pretrained_model(xs).numpy())), axis=0)
    return feat

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    #mean = np.array([0.485, 0.456, 0.406])
    #std = np.array([0.229, 0.224, 0.225])
    #inp = std * inp + mean
    #inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)# pause a bit so that plots are updated

############# prepare data ############

size = 224

img=generate_DataSet(data_List,img_root)
DataSet=get_dataset_cropped(img,size)
X,y,group = get_data_ready(DataSet)
y = y.astype(int)

############ visualize #################

Xgray = X[:,:,:,0]*0.299 + X[:,:,:,1]*0.587 + X[:,:,:,2]*0.114
mean=np.mean(Xgray,axis=())
stds = np.std( Xgray, axis = (1,2) )
filt = stds>17

'''
plt.rcParams['figure.figsize'] = (5,5)
plt.hist(stds, bins=50)  # arguments are passed to np.histogram
plt.title("Histogram")
plt.show()


plt.rcParams['figure.figsize'] = (20,60)
out = torchvision.utils.make_grid(torch.from_numpy(X[filt].transpose(0,3,1,2)),nrow = 20 )
print(len(X[filt]))
imshow(out)
plt.show()
'''

############### main ########################

X_feat_res = generate_feature(X)

f = open('features_train%d.pkl','wb')
pickle.dump(X_feat_res, f)
f.close()

f = open('label_train%d.pkl','wb')
pickle.dump(y, f)
f.close()


