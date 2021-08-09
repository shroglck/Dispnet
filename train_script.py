import torch
import os
import torch.utils.data as data
import torchvision.transforms as transforms
import random
from PIL import Image,ImageOps
import numpy as np
from datetime import datetime
import torch.nn as nn
import torch.functional as F
import math
from prototype import Encoder_Decoder
from dataloader import SceneData
from losses import smoothness,perceptual_loss,deep_correlation_loss
import closed_form_matting
model=Encoder_Decoder().cuda()
data=SceneData()
train_loader=torch.utils.data.DataLoader(dataset=data,batch_size=4,shuffle=True)
epochs=1
optimizer=torch.optim.Adam(model.parameters(),lr=0.0001)

def main():
    train_losses=[]
    for i in range(epochs):
        t0=datetime.now()
        train_loss = train(train_loader,model,optimizer)
        train_losses.append(train_loss)
        dt=datetime.now()-t0
        print(f'Epoch {i+1}/{epochs}, Train Loss: {train_loss:.4f}, Duration: {dt}')
        optimizer.step()
        #save data
def train(data, model,optimizer):
    model.train(True)
    losses=0
    cnt=0
    for left,right,disp,alpha  in data:
        cnt+=1
        left=left.cuda()
        right=right.cuda()
        B,C,H,W= left.shape
        optimizer.zero_grad()
        output=model(left,right)
        
        alpha=alpha.cuda()
        alpha=alpha.unsqueeze(1)
        output=output*alpha
        avg_layer=torch.nn.AvgPool2d(kernel_size=5,stride=1,padding=2).cuda()
        output=avg_layer(output)
        disp=disp.unsqueeze(1)
        sm_loss= smoothness(left[:, :, :, int(0.20 * 256)::], output[:, :, :, int(0.20 * 256)::], gamma=2).cpu()
        sm_loss=(2*.2/512)*sm_loss
        loss=sm_loss
        output=output.cpu()
        loss=loss+.01*deep_correlation_loss(output,disp)
        loss+= .1*torch.nn.L1Loss()(disp,output)
        loss.backward()
        losses=loss.item()+losses
        optimizer.step()
        optimizer.zero_grad()
        # add matting adn deep correlation loss
    return losses/cnt

def save():
    path="trial.pth"
    torch.save(model.state_dict(),path)

if __name__=='__main__':
    main()
    save()
    print(model.state_dict())

