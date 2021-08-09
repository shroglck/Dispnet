import torch
import os
import torch.utils.data as data
import torchvision.transforms as transforms
import random
from PIL import Image,ImageOps
import numpy as np
import preprocess
import os 
import os.path
import closed_form_matting

def default_loader(path):
    return Image.open(path).convert('RGB')

def disparity_loader(path):
    return Image.open(path)


class Kitti_loader(data.Dataset):
    def __init__(self, left, right, left_disparity, training, loader=default_loader, dploader= disparity_loader):
        super(Kitti_loader,self).__init__()
        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training

    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]
        disp_L= self.disp_L[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL = self.dploader(disp_L)


        if self.training:  
            w, h = left_img.size
            th, tw = 128, 256
 
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

            dataL = np.ascontiguousarray(dataL,dtype=np.float32)/256
            dataL = dataL[y1:y1 + th, x1:x1 + tw]

            processed = preprocess.get_transform(augment=False)  
            left_img   = processed(left_img)
            right_img  = processed(right_img)
            l_img=left_img.permute(1,2,0)
            alpha= closed_form_matting.closed_form_matting_with_trimap(np.array(l_img),np.array(dataL))
            alpha=np.float32(alpha)
            return left_img, right_img, dataL,alpha
        else:
            w, h = left_img.size

            left_img = left_img.crop((w-1232, h-368, w, h))
            right_img = right_img.crop((w-1232, h-368, w, h))
            w1, h1 = left_img.size

            dataL = dataL.crop((w-1232, h-368, w, h))
            dataL = np.ascontiguousarray(dataL,dtype=np.float32)/256

            processed = preprocess.get_transform(augment=False)  
            left_img       = processed(left_img)
            right_img      = processed(right_img)
            l_img=left_img.permute(1,2,0)
            alpha= closed_form_matting.closed_form_matting_with_trimap(np.array(l_img),np.array(dataL))
            alpha=np.float32(alpha)

            return left_img, right_img, dataL,alpha

    def __len__(self):
        return len(self.left)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
def Kitti():
    file_path= "/home/nitik/ICRA_paper/data_scene_flow/training/"
    left_fold  = 'image_2/'
    right_fold = 'image_3/'
    disp_L = 'disp_occ_0/'
    left_files=file_path+"/"+left_fold
    right_files=file_path+"/"+right_fold
    disparity_files=file_path+"/"+disp_L
    image = [img for img in os.listdir(left_files) if img.find('_10') > -1]
    left_directory = [left_files+file for file in image]
    right_directory = [right_files+file for file in image]
    disparity_directory=[disparity_files+file for file in image]

    Trian=Kitti_loader(left=left_directory,right=right_directory,left_disparity=disparity_directory,training=True)
    return Trian
