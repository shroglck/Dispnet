import torch
import os
import torch.utils.data as data
import torchvision.transforms as transforms
import random
from PIL import Image,ImageOps
import numpy as np
import torch.nn as nn
import torch.functional as F
import math

class SelfRouting2d(nn.Module):
    def __init__(self, A, B, C, D, kernel_size=3, stride=1, padding=1, pose_out=False):
        super(SelfRouting2d, self).__init__()
        self.A = A
        self.B = B
        self.C = C
        self.D = D

        self.k = 3
        self.kk = 9
        self.kkA = self.kk * A

        self.stride = 1
        self.pad = 1

        self.pose_out = pose_out

        if pose_out:
            self.W1 = nn.Parameter(torch.FloatTensor(self.kkA, B*D, C))
            nn.init.kaiming_uniform_(self.W1.data)

        self.W2 = nn.Parameter(torch.FloatTensor(self.kkA, B, C))
        self.b2 = nn.Parameter(torch.FloatTensor(1, 1, self.kkA, B))

        nn.init.constant_(self.W2.data, 0)
        nn.init.constant_(self.b2.data, 0)

    def forward(self, a, pose):
        # a: [b, A, h, w]
        # pose: [b, AC, h, w]
        b, _, h, w = a.shape
        # [b, ACkk, l]
        pose = torch.nn.functional.unfold(pose, 3,1,1)
        
        l = pose.shape[-1]
        # [b, A, C, kk, l]
        pose = pose.view(b, self.A,self.C,self.kk, l)
        # [b, l, kk, A, C]
        pose = pose.permute(0, 4, 3, 1, 2).contiguous()
        # [b, l, kkA, C, 1]
        pose = pose.view(b, l, self.kkA, self.C, 1)
        if hasattr(self, 'W1'):
            # [b, l, kkA, BD]
            pose_out = torch.matmul(self.W1, pose).squeeze(-1)
            # [b, l, kkA, B, D]
            pose_out = pose_out.view(b, l, self.kkA, self.B, self.D)

        # [b, l, kkA, B]
        logit = torch.matmul(self.W2, pose).squeeze(-1) + self.b2

        # [b, l, kkA, B]
        r = torch.softmax(logit, dim=3)

        # [b, kkA, l]
        a = torch.nn.functional.unfold(a, self.k, stride=self.stride, padding=self.pad)
        # [b, A, kk, l]
        a = a.view(b, self.A, self.kk, l)
        # [b, l, kk, A]
        a = a.permute(0, 3, 2, 1).contiguous()
        # [b, l, kkA, 1]
        a = a.view(b, l, self.kkA, 1)

        # [b, l, kkA, B]
        ar = a * r
        # [b, l, 1, B]
        ar_sum = ar.sum(dim=2, keepdim=True)
        # [b, l, kkA, B, 1]
        coeff = (ar / (ar_sum)).unsqueeze(-1)

        # [b, l, B]
        # a_out = ar_sum.squeeze(2)
        a_out = ar_sum / a.sum(dim=2, keepdim=True)
        a_out = a_out.squeeze(2)

        # [b, B, l]
        a_out = a_out.transpose(1,2)
        
        if hasattr(self, 'W1'):
            # [b, l, B, D]
            pose_out = (coeff * pose_out).sum(dim=2)
            # [b, l, BD]
            pose_out = pose_out.view(b, l, -1)
            # [b, BD, l]
            pose_out = pose_out.transpose(1,2)
          

        oh = ow = math.floor(l**(1/2))
        a_out = a_out.view(b, -1, h, w)
        if hasattr(self, 'W1'):
            pose_out = pose_out.view(b, -1, h, w)
        else:
            pose_out = None

        return a_out, pose_out

def squash(s, dim=-1):
    mag_sq = torch.sum(s**2, dim=dim, keepdim=True)
    mag = torch.sqrt(mag_sq)
    v = (mag_sq / (1.0 + mag_sq)) * (s / mag)
    return v

def weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class ConvNet(nn.Module):
    def __init__(self, planes, cfg_data, num_caps, caps_size, depth, mode):
        caps_size = 16
        super(ConvNet, self).__init__()
        channels, classes = cfg_data['channels'], cfg_data['classes']
        self.num_caps = num_caps
        self.caps_size = caps_size
        self.depth = depth
        self.mode = mode

        self.layers = nn.Sequential(
            nn.Conv2d(channels, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            nn.Conv2d(planes, planes*2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes*2),
            nn.ReLU(True),
            nn.Conv2d(planes*2, planes*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes*4),
            nn.ReLU(True),)
            

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.conv_layers.append(SelfRouting2d(4,2, caps_size, caps_size, kernel_size=3, stride=1, padding=1, pose_out=True))
        self.conv_layers.append(SelfRouting2d(2,2, caps_size, caps_size, kernel_size=3, stride=1, padding=1, pose_out=True))
        self.norm_layers.append(nn.BatchNorm2d(caps_size*2))
        self.norm_layers.append(nn.BatchNorm2d(caps_size*2))
           
        final_shape = 4

        self.conv_a = nn.Conv2d(4*planes, num_caps, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_pose = nn.Conv2d(4*planes, num_caps*caps_size, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(num_caps)
        self.bn_pose = nn.BatchNorm2d(num_caps*caps_size)
        self.fc = SelfRouting2d(2, classes, caps_size, 1, kernel_size=final_shape, padding=0, pose_out=False)

        self.apply(weights_init)

    def forward(self, x):
        out = self.layers(x)
        # DR
        if self.mode == 'SR':
            a, pose = self.conv_a(out), self.conv_pose(out)
            a, pose = torch.sigmoid(self.bn_a(a)), self.bn_pose(pose)
            for m, bn in zip(self.conv_layers, self.norm_layers):
                a, pose = m(a, pose)
                pose = bn(pose)
                
            a, _ = self.fc(a, pose)
            a=a.contiguous()
            out = a.view(a.size(0), -1)
            out = out.log()

        
        return out
