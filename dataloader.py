import torch
import os
import torch.utils.data as data
import torchvision.transforms as transforms
import random
from PIL import Image,ImageOps
import readpfm as rp
import numpy as np
import preprocess
import os 
import os.path
import closed_form_matting



def default_loader (path):
    return Image.open(path).convert('RGB')

def pfm_loader (path):
    return rp.readPFM(path)

class scene_flow_loader(data.Dataset):
    def __init__(self, left, right, left_disparity,training,loader=default_loader, dp_loader=pfm_loader):
        super(scene_flow_loader,self).__init__()

        self.left=left
        self.right= right
        self.dispL=left_disparity
        self.loader=loader
        self.dp_loader=dp_loader
        self.training=training
    
    def __getitem__(self,index):
        left=self.left[index]
        right=self.right[index]
        disp_L=self.dispL[index]

        left_img=self.loader(left)
        right_img=self.loader(right)
        dataL, scaleL= self.dp_loader(disp_L)
        dataL = np.ascontiguousarray(dataL,dtype=np.float32)
        
        if self.training:
            w,h = left_img.size
            th,tw= 128,256

            x1=random.randint(0,w-tw)
            y1=random.randint(0,h-th)

            left_img=left_img.crop((x1,y1,x1+tw,y1+th))
            right_img = right_img.crop((x1,y1,x1+tw,y1+th))

            dataL=dataL[y1:y1+th,x1:x1+tw]
            processed = preprocess.get_transform(augment=False)  
            left_img   = processed(left_img)
            right_img  = processed(right_img)
            l_img=left_img.permute(1,2,0)
            alpha=closed_form_matting.closed_form_matting_with_trimap(np.array(l_img),dataL)
            alpha=np.float32(alpha)
            return left_img,right_img,dataL,alpha
        else:
            processed = preprocess.get_transform(augment=False)  
            left_img       = processed(left_img)
            right_img      = processed(right_img)
            alpha=closed_form_matting.closed_form_matting_with_trimap(left_img,dataL)
            return left_img,right_img,dataL
    
    def __len__(self):
        return (len(self.left))

def SceneData():
    file_path="/home/nitik/ICRA_paper/data"
    left_files=file_path+"/left"
    right_files=file_path+"/right"
    disparity_files=file_path+"/disparity"

    left_directory = [left_files+"/"+file for file in os.listdir(left_files)]
    right_directory = [right_files+"/"+file for file in os.listdir(right_files)]
    disparity_directory=[disparity_files+"/"+file for file in os.listdir(disparity_files)]

    Trian=scene_flow_loader(left=left_directory,right=right_directory,left_disparity=disparity_directory,training=True)
    return Trian
