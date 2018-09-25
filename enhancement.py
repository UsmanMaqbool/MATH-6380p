from PIL import ImageEnhance
from PIL import Image
import os
import pickle

file_name=list()
img_enhance =dict()
path = '/home/ylan/下载/Pictures'

def read_name (path):
    return os.listdir(path)

def enhance (path):

    file_name = read_name(path)
    for i in file_name:
        print(path+'/'+ i)
        img = Image.open(path+'/'+ i)
        img_=ImageEnhance.Contrast(img).enhance(1.2)
        img_enhance[i]=img_

    return img_enhance

imgenhance= enhance(path)

f = open('enhance_img.pkl','w')
pickle.dump(imgenhance, f)
f.close()




