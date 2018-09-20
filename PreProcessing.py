from PIL import Image
import numpy as np
import os

def readImg(path,imgs):
    ims = {}
    for img in imgs:
        try:
            im = Image.open(path+"/"+img)
            ims[img] = im
        except Exception as Error:
            print(img+"Error")
    return ims
def read_raw_dataset(root_path):
    outputpath = root_path
    classes = ["0","1"]
    raw_img = {}
    for cls in classes:
        path = outputpath+"/"+cls
        imgs = np.sort( os.listdir(path) )
        imgs_jpg = np.char.find(imgs,".jpg")>0
        raw_img[cls] = readImg(path,imgs[imgs_jpg])
    return raw_img
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
def get_multi_scale_dataset(raw_img,scales,size):
    ratio = scales
    dataset = {}
    for cls in raw_img:
        dataset[cls] = {}
        for im in raw_img[cls]:
            dataset[cls][im] = {}
            for rt in ratio:
                resized_im = resize_img(raw_img[cls][im],rt)
                dataset[cls][im][rt] = crop_and_resize(resized_im,size)
    return dataset
    
    
    
scales = [1,0.5,0.25,0.125]
size = 224
path = '/Users/scraed/Downloads/Raphael-jpg'
raw_img = read_raw_dataset(path)   
dataset = get_multi_scale_dataset(raw_img,scales,size)

# dataset structure:
# class: ["0","1"],filename: ["xxx.jpg"] scale: []
# Example: dataset["1"]['20_1.jpg'][0.5]  is a (280, 224, 224, 3) numpy array, 280 pieces of 224*224 picture chunks
