import torch
import os
import torch.utils.data as data
import torchvision.transforms as transforms
import random
from PIL import Image,ImageOps
import readpfm as rp
import numpy as np
import preprocess
import torch.nn as nn
import torch.functional as F
from  dataloader import SceneData
from capsulenetwork import SelfRouting2d


def conv_activation(in_planes,num_capsule):
    return nn.Conv2d(in_channels=in_planes,out_channels=num_capsule,
    kernel_size=3,stride=1,padding=1,bias=False)

def conv_pose(in_planes,num_capsule,capsule_length):
    return nn.Conv2d(in_channels=in_planes,out_channels=num_capsule*capsule_length,
    kernel_size=3,stride=1,padding=1,bias=False)


class residual_block(nn.Module):
    def __init__(self, in_planes, kernel_size=3):
        super(residual_block, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=(kernel_size, 1), padding=((kernel_size - 1) // 2, 0),
                               bias=False)
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=(1, kernel_size), padding=(0, (kernel_size - 1) // 2),
                               bias=False)

    def forward(self, x):
        x = self.elu(self.conv2(self.elu(self.conv1(x))) + x)
        return x

def conv_elu(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, pad=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=pad,
                      bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ELU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=pad,
                      bias=True),
            nn.ELU(inplace=True)
        )

class deconv(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(deconv, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x, ref):
        x = torch.nn.functional.interpolate(x, size=(ref.size(2), ref.size(3)), mode='nearest')
        x = self.elu(self.conv1(x))
        return x

def resolve_layer(num_channels,out_channels):
    return nn.Conv2d(in_channels=num_channels,out_channels=out_channels,kernel_size=1,stride=1,padding =0)


class Encoder_Decoder(nn.Module):
    def __init__(self,batch_norm=True,in_channels=3,out_channels=1,planes=64,num_caps=8,caps_size=3,num_of_levels=4,no_fac=1):
        super(Encoder_Decoder,self).__init__()
        self.num_of_levels=num_of_levels
        self.no_fac=no_fac
        self.batch_norm=batch_norm
        self.conv0 = conv_elu(batch_norm,3,64,9,1,4)
        self.res0=residual_block(in_planes=64)
        self.resolve0= resolve_layer(128,64)
        self.conv_a0=conv_activation(64,4)
        self.conv_p0=conv_pose(64,4,caps_size)
        self.bn_a0=nn.BatchNorm2d(4)
        self.bn_p0=nn.BatchNorm2d(4*caps_size)
        self.caps0=SelfRouting2d(4,4,caps_size,caps_size, kernel_size=3, stride=1, padding=1, pose_out=True)
        self.bn_p1=nn.BatchNorm2d(4*caps_size)
        

        self.conv1 = conv_elu(batch_norm,64,128,7,1,3)
        self.res1=residual_block(in_planes=128)
        self.resolve1= resolve_layer(128+4,128)
        self.conv_a1=conv_activation(128,4)
        self.bn_a1=nn.BatchNorm2d(4)
        self.caps1=SelfRouting2d(4,4,caps_size,caps_size, kernel_size=3, stride=1, padding=1, pose_out=True)
        self.bn_p2=nn.BatchNorm2d(4*caps_size)
        
        self.conv2=conv_elu(batch_norm,128,256,5,1,pad=2)
        self.res2=residual_block(in_planes=256)
        self.resolve2=resolve_layer(256+4,256)
        self.conv_a2=conv_activation(256,4)
        self.bn_a2=nn.BatchNorm2d(4)
        self.caps2=SelfRouting2d(4,4,caps_size,caps_size,kernel_size=3, stride=1,padding=1, pose_out=True)
        self.bn_p3=nn.BatchNorm2d(4*caps_size)
        

        self.conv3=conv_elu(batch_norm,256,256,3,1,pad=1)
        self.res3=residual_block(in_planes=256)
        self.resolve3=resolve_layer(256+4,256)
        self.conv_a3=conv_activation(256,4)
        self.bn_a3=nn.BatchNorm2d(4)
        self.caps3=SelfRouting2d(4,4,caps_size,caps_size,kernel_size=3, stride=1,padding=1, pose_out=False)
        
        
        
        
        
        
        
        
        
        
        
    def forward(self,x,y):
        b,_,h,w=x.shape

        out_x1=self.res0(self.conv0(x))
        out_1=out_x1

        out_a1,out_p1=self.conv_a0(out_1),self.conv_p0(out_1)
        out_a1=torch.sigmoid(self.bn_a0(out_a1))
       
        out_p1=self.bn_p0(out_p1)

        out_a1,out_p1=self.caps0(out_a1,out_p1)
        out_p1=self.bn_p1(out_p1)

        ## start of conv 2
        out_x1=self.res1(self.conv1(out_x1))
        out_1= torch.cat((out_x1,out_a1),1)
        out_1=self.resolve1(out_1)
        out_a1=self.conv_a1(out_1)
        out_a1=torch.sigmoid(self.bn_a1(out_a1))
        
        out_a1,out_p1=self.caps1(out_a1,out_p1)
        out_p1=self.bn_p1(out_p1)

        ## star of conv 3

        out_x1=self.res2(self.conv2(out_x1))
        out_1= torch.cat((out_x1,out_a1),1)
        out_1=self.resolve2(out_1)
        out_a1=self.conv_a2(out_1)
        out_a1=torch.sigmoid(self.bn_a2(out_a1))
        
        out_a1,out_p1=self.caps2(out_a1,out_p1)
        out_p1=self.bn_p2(out_p1)

        ## start of conv 4##

        out_x1=self.res3(self.conv3(out_x1))
        out_1= torch.cat((out_x1,out_a1),1)
        out_1=self.resolve3(out_1)
        out_a1=self.conv_a3(out_1)
        out_a1=torch.sigmoid(self.bn_a3(out_a1))
        out_a1,_=self.caps3(out_a1,out_p1)
        

        a=out_a1
        a=a.contiguous()
        sout=torch.softmax(a,dim=1)
        disp=0
        max_disp=torch.tensor([300]).unsqueeze(1).unsqueeze(1).type(x.type())
        min_disp=torch.tensor([3]).unsqueeze(1).unsqueeze(1).type(x.type())
        for n in range(0, 4):
            with torch.no_grad():
                    # Exponential quantization
                c = n / (self.num_of_levels * self.no_fac - 1)  # Goes from 0 to 1
                w = 300 * torch.exp(torch.log(max_disp /min_disp) * (c - 1))
            disp = disp + w.unsqueeze(1) * sout[:, n, :, :].unsqueeze(1)
        
        return disp


