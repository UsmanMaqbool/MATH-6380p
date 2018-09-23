from PIL import Image
import numpy as np
import os
import cv2
from scatnetgpu import ScatNet, stack_scat_output
import glob
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
# scales = [1]

size = 224
path = '/app/scanet/Raphael-jpg'
raw_img = read_raw_dataset(path)   
dataset = get_multi_scale_dataset(raw_img,scales,size)
# print("dataset {}".format(dataset))
# print("dataset.keys {}".format(dataset.keys()))
# print("dataset[1][20_1.jpg][0.5][0,:,:,:] shape: {}".format(dataset["1"]['20_1.jpg'][0.5][0,:,:,:].shape))
# img = dataset["1"]['20_1.jpg'][0.5][0,:,:,:]
# print("type of img is {}".format(type(img)))
prefix = "/app/scanet/Raphael-jpg/"
# cv2.imshow("img", img)
# cv2.waitKey(0)
folder_list = ["0", "1"]
sn = ScatNet(M=2, J=4, L=6)
for folder in folder_list:
    for img_name in glob.glob(os.path.join(prefix, folder, "*")):
        img_name = img_name.split("/")[-1]
        for scale in scales:
            print("folder: {}, img: {}, scale: {}".format(folder, img_name, scale))
            imgs = dataset[folder][img_name][scale][:,:,:,:]
            out = sn.transform_batch(imgs)
            print("type of out is : {}".format(type(out)))
            print("len of out is : {}".format(len(out)))
            feats0, feats1, feats2 = [], [], []
            for idx in range(len(out)):
                feat0, _ = stack_scat_output(out[idx][0])
                feat0 = feat0.flatten()                         # To make it one array
                feats0.append(feat0)                            # Attach to the feats back
                feat1, _ = stack_scat_output(out[idx][1])
                feat1 = feat1.flatten()
                feats1.append(feat1)
                feat2, _ = stack_scat_output(out[idx][2])
                feat2 = feat2.flatten()
                feats2.append(feat2)
            print("feats0 : {}".format(len(feats0)))
            print("feats1 : {}".format(len(feats1)))
            print("feats2 : {}".format(len(feats2)))
            feat_im0 = np.concatenate(feats0)
            feat_im1 = np.concatenate(feats1)
            feat_im2 = np.concatenate(feats2)
            feat_im = np.concatenate([feat_im0, feat_im1, feat_im2])
            print("feat_im0 : {}".format(feat_im0.shape))   # Channel R
            print("feat_im1 : {}".format(feat_im1.shape))   # Channel G
            print("feat_im2 : {}".format(feat_im2.shape))   # Channel B
            print("feat_im : {}".format(feat_im.shape))     # Channel All
            print("feat_im lenth : {}".format(len(feat_im)))     # Channel All
            img_name_base = os.path.splitext(img_name)[0]
            print("img_name_base : {}".format(img_name_base))     # ImageNAme
            if scale == 1:
                np.save(os.path.join('features', img_name_base + '1'), feat_im)
            elif scale == 0.5:
                np.save(os.path.join('features', img_name_base + '5'), feat_im)
            elif scale == 0.25:
                np.save(os.path.join('features', img_name_base + '25'), feat_im)
            elif scale == 0.125:
                np.save(os.path.join('features', img_name_base + '125'), feat_im)
            
            # print("len of feat0 is {}".format(feat0.shape))
            # print("len of feat1 is {}".format(feat1.shape))
            # print("len of feat2 is {}".format(feat2.shape))
            #             
            # for idx in range(dataset[folder][img_name][scale].shape[0]):
            #     print("folder: {}, img: {}, scale: {}, idx: {}".format(folder, img_name, scale, idx))
            #     img = dataset[folder][img_name][scale][idx,:,:,:]
            #     sn = ScatNet(M=2, J=4, L=6)
            #     out = sn.transform(img)
            #     feat0, _ = stack_scat_output(out[0])
            #     feat0 = feat0.flatten()
            #     feat1, _ = stack_scat_output(out[1])
            #     feat1 = feat1.flatten()
            #     feat2, _ = stack_scat_output(out[2])
            #     feat2 = feat2.flatten()
            #     print("len of feat0 is {}".format(feat0.shape))
            #     print("len of feat1 is {}".format(feat1.shape))
            #     print("len of feat2 is {}".format(feat2.shape))



# out = sn.transform(img)
# dataset structure:
# class: ["0","1"],filename: ["xxx.jpg"] scale: []
# Example: dataset["1"]['20_1.jpg'][0.5]  is a (280, 224, 224, 3) numpy array, 280 pieces of 224*224 picture chunks

