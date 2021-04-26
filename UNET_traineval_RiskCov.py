"""
Created on Tue Mar 23 11:35:53 2021

@author: michalablicher
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Defines the Unet.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1 at the bottleneck

#%% Load packages
import torch
import os
#import cv2
import nibabel as nib
import numpy   as np
#import pandas  as pd
import matplotlib.pyplot as plt
import torchvision
import glob2
#import time

#from skimage.transform import resize
from torch import nn
from torch import Tensor

import scipy


#!pip install torch-summary
#!pip install opencv-python

#%% BayesUNet
# recursive implementation of Unet

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
    
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.zero_()


class UNet(nn.Module):
    def __init__(self, num_classes=3, in_channels=1, initial_filter_size=64, kernel_size=3, num_downs=4,
                 norm_layer=nn.InstanceNorm2d, drop_prob=0.):
        super(UNet, self).__init__()
        self.drop_prob = drop_prob
        # construct UNet structure
        unet_block = UnetSkipConnectionBlock(in_channels=initial_filter_size * 2 ** (num_downs-1), out_channels=initial_filter_size * 2 ** num_downs,
                                             num_classes=num_classes, kernel_size=kernel_size, norm_layer=norm_layer,
                                             innermost=True, drop_prob=self.drop_prob)
        for i in range(1, num_downs):
            unet_block = UnetSkipConnectionBlock(in_channels=initial_filter_size * 2 ** (num_downs-(i+1)),
                                                 out_channels=initial_filter_size * 2 ** (num_downs-i),
                                                 num_classes=num_classes, kernel_size=kernel_size, submodule=unet_block,
                                                 norm_layer=norm_layer, drop_prob=self.drop_prob
                                                 )
            
        unet_block = UnetSkipConnectionBlock(in_channels=in_channels, out_channels=initial_filter_size,
                                             num_classes=num_classes, kernel_size=kernel_size, submodule=unet_block, norm_layer=norm_layer,
                                             outermost=True, drop_prob=self.drop_prob)

        self.model = unet_block
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.apply(weights_init)

    def forward(self, x):
        out = self.model(x)
        return {'log_softmax': self.log_softmax(out), 'softmax': self.softmax(out)}


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, num_classes=1, kernel_size=3,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.InstanceNorm2d, drop_prob=0.):
        super(UnetSkipConnectionBlock, self).__init__()
        self.use_dropout = True if drop_prob > 0. else False
        self.drop_prob = drop_prob
        self.outermost = outermost
        # downconv
        pool = nn.MaxPool2d(2, stride=2)
        conv1 = self.contract(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, norm_layer=norm_layer)
        conv2 = self.contract(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, norm_layer=norm_layer)

        # upconv
        conv3 = self.expand(in_channels=out_channels*2, out_channels=out_channels, kernel_size=kernel_size)
        conv4 = self.expand(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size)

        if outermost:
            final = nn.Conv2d(out_channels, num_classes, kernel_size=1)
            if self.use_dropout:
                down = [conv1, conv2, nn.Dropout2d(self.drop_prob)]
            else:
                down = [conv1, conv2]
            if self.use_dropout:
                up = [conv3, nn.Dropout2d(self.drop_prob), conv4, nn.Dropout2d(self.drop_prob), final]
            else:
                up = [conv3, conv4, final]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(in_channels*2, in_channels,
                                        kernel_size=2, stride=2)
            model = [pool, conv1, conv2, upconv]
        else:
            upconv = nn.ConvTranspose2d(in_channels*2, in_channels, kernel_size=2, stride=2)

            down = [pool, conv1, conv2]
            up = [conv3, conv4, upconv]

            if self.use_dropout:
                model = down + [nn.Dropout2d(self.drop_prob)] + [submodule] + up + [nn.Dropout2d(self.drop_prob)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    @staticmethod
    def contract(in_channels, out_channels, kernel_size=3, norm_layer=nn.InstanceNorm2d):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            norm_layer(out_channels),
            nn.LeakyReLU(inplace=True))
        return layer

    @staticmethod
    def expand(in_channels, out_channels, kernel_size=3):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        return layer

    @staticmethod
    def center_crop(layer, target_width, target_height):
        batch_size, n_channels, layer_width, layer_height = layer.size()

        xy1 = (layer_width - target_width) // 2
        xy2 = (layer_height - target_height) // 2
        return layer[:, :, xy1:(xy1 + target_width), xy2:(xy2 + target_height)]

    def forward(self, x):
        if self.outermost:
            out = self.model(x)
            return out
        else:
            crop = self.center_crop(self.model(x), x.size()[2], x.size()[3])
            out = torch.cat([x, crop], 1)
            return out


class BayesUNet(UNet):

    def __init__(self, num_classes=3, in_channels=1, initial_filter_size=64, kernel_size=3, num_downs=4,
                 norm_layer=nn.InstanceNorm2d, drop_prob=0.):
        super(BayesUNet, self).__init__(num_classes, in_channels, initial_filter_size, kernel_size, num_downs,
                 norm_layer=norm_layer, drop_prob=drop_prob)

    def train(self, mode=True, mc_dropout=False):
        """ Sets the module in training mode.
            !!! OVERWRITING STANDARD PYTORCH METHOD for nn.Module

            OR
                if mc_dropout=True and mode=False (use dropout during inference) we set all modules
                to train-mode=False except for DROPOUT layers
                In this case it is important that the module_name matches BayesDRNSeg.dropout_layer

        Returns:
            Module: self
        """
        self.training = mode
        for module_name, module in self.named_modules():
            module.training = mode
            if mc_dropout and not mode:
                if isinstance(module, nn.Dropout2d):
                    # print("WARNING - nn.Module.train - {}".format(module_name))
                    module.training = True

        return self

    def eval(self, mc_dropout=False):
        """Sets the module in evaluation mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.
        """
        return self.train(False, mc_dropout=mc_dropout)

if __name__ == "__main__":
    #import torchsummary
    model = BayesUNet(num_classes=4, in_channels=1, drop_prob=0.1)
    #model.cuda()
    #torchsummary.summary(model, (1, 128, 128))

#%% Load image  
cwd = os.getcwd()
os.chdir("C:/Users/katrine/Documents/Universitet/Speciale/ACDC_training_data/training")
#os.chdir('/Users/michalablicher/Desktop/training')

frame_dia_im = np.sort(glob2.glob('patient*/**/patient*_frame01.nii.gz'))

num_patients = len(frame_dia_im)
H = 128
W = 128
in_c = 1

data_im = []
centercrop     = torchvision.transforms.CenterCrop((H,W))

for i in range(0,num_patients):
    nimg = nib.load(frame_dia_im[i])
    img  = nimg.get_fdata()
    
    im_slices      = img.shape[2]
    centercrop_img = Tensor(np.zeros((H,W,im_slices)))
    
    for j in range(0,im_slices):
        centercrop_img[:,:,j] = centercrop(Tensor(img[:,:,j]))
   
    in_image = np.expand_dims(centercrop_img,0)
    in_image = Tensor(in_image).permute(3,0,1,2).detach().numpy()
    
    data_im.append(in_image.astype(object))

#%% Load annotations
frame_dia_gt = np.sort(glob2.glob('patient*/**/patient*_frame01_gt.nii.gz'))

num_patients = len(frame_dia_gt)
H = 128
W = 128


data_gt = [] 
centercrop     = torchvision.transforms.CenterCrop((H,W))

for i in range(0,num_patients):
    n_gt = nib.load(frame_dia_gt[i])
    gt  = n_gt.get_fdata()
    
    gt_slices      = gt.shape[2]
    centercrop_gt = Tensor(np.zeros((H,W,gt_slices)))
    
    for j in range(0,gt_slices):
        centercrop_gt[:,:,j] = centercrop(Tensor(gt[:,:,j]))
   
    in_gt = Tensor(centercrop_gt).permute(2,0,1).detach().numpy()
    
    data_gt.append(in_gt.astype(object))

#%% BATCH GENERATOR

num = 5

num_train = num 
num_eval  = num + num_train 
num_test  = num + num_eval

im_flat_train = np.concatenate(data_im[0:num_train]).astype(None)
gt_flat_train = np.concatenate(data_gt[0:num_train]).astype(None)

im_flat_eval = np.concatenate(data_im[num_train:num_eval]).astype(None)
gt_flat_eval = np.concatenate(data_gt[num_train:num_eval]).astype(None)

im_flat_test = np.concatenate(data_im[num_eval:num_test]).astype(None)
gt_flat_test = np.concatenate(data_gt[num_eval:num_test]).astype(None)

#%% Setting up training loop
# OBS DECREASED LEARNING RATE AND EPSILON ADDED TO OPTIMIZER

import torch.optim as optim
from torch.autograd  import Variable
#from sklearn.metrics import brier_score_loss

LEARNING_RATE = 0.0001 # 
criterion    = nn.CrossEntropyLoss() 
#criterion     = nn.BCELoss()
#criterion     = SoftDice
#criterion     = brier_score_loss()

# weight_decay is equal to L2 regularizationst
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-04, weight_decay=1e-4)
# torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

# and a learning rate scheduler which decreases the learning rate by 10x every 3 epochs
#lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
#                                               step_size=3,
#                                               gamma=0.1)

num_epoch = 10

#%% Training
losses = []
losses_eval = []
trainloader = im_flat_train

for epoch in range(num_epoch):  # loop over the dataset multiple times
    
    model.train()
    print('Epoch train =',epoch)
    train_loss = 0.0  
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        #inputs, labels = data
        inputs = Tensor(im_flat_train)
        labels = Tensor(gt_flat_train)
        #print('i=',i)

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)
        labels = labels.long()

        # Clear the gradients
        optimizer.zero_grad()

        # Forward Pass
        output = model(inputs)     
        output = output["log_softmax"]
        
        # Find loss
        loss = criterion(output, labels)
        # Calculate gradients
        loss.backward()
        # Update Weights
        optimizer.step()
        
        # Calculate loss
        train_loss += loss.item()
    losses.append(train_loss/trainloader.shape[0]) # This should be normalised by batch size
    train_loss = 0.0
     
    model.eval()
    print('Epoch eval=',epoch)
    eval_loss = 0.0  
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        #inputs, labels = data
        inputs = Tensor(im_flat_eval)
        labels = Tensor(gt_flat_eval)
        #print('i=',i)

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)
        labels = labels.long()
        
        # Forward pass
        output = model(inputs)     
        output = output["log_softmax"]
        # Find loss
        loss = criterion(output, labels)

        # Calculate loss
        eval_loss += loss.item()
    losses_eval.append(eval_loss/trainloader.shape[0])
        #(epoch + 1, i + 1, eval_loss)
    eval_loss = 0.0

print('Finished Training + Evaluation')
        
#%% Plot loss curves

epochs = np.arange(len(losses))
epochs_eval = np.arange(len(losses_eval))

plt.figure(dpi=200)
plt.plot(epochs+1, losses, 'b', label='Training Loss')
plt.plot(epochs_eval+1, losses_eval, 'r', label='Validation Loss')
plt.xticks(np.arange(1,11, step=1))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc="upper right")
plt.title("Loss function")
plt.show()

#%% Save model
PATH_model = "C:/Users/katrine/Documents/Universitet/Speciale/trained_Unet_locally.pt"
PATH_state = "C:/Users/katrine/Documents/Universitet/Speciale/trained_Unet_locallystate.pt"
torch.save(model, PATH_model)
torch.save(model.state_dict(), PATH_state)


#%% Load model
PATH_model = "C:/Users/katrine/Documents/Universitet/Speciale/Trained models/trained_Unet_locally.pt"
#PATH_state = "C:/Users/katrine/Documents/Universitet/Speciale/Trained models/trained_Unet_locallystate.pt"

# Load
model = torch.load(PATH_model)
#model.load_state_dict(torch.load(PATH_state))
model.eval()


#%% TESTING! 
# Load new image for testing
cwd = os.getcwd()
os.chdir("C:/Users/katrine/Documents/Universitet/Speciale/ACDC_training_data/training/patient051")
#os.chdir('/Users/michalablicher/Desktop/training/patient009')

nimg = nib.load('patient051_frame01.nii.gz')
img  = nimg.get_fdata()

im_slices = img.shape[2]

for i in range(0,im_slices):
    plt.suptitle('Test image slices')
    plt.subplot(2,5,i+1)
    plt.imshow(img[:,:,i])    
    
#%% Prepare image and run through model
# Crop image
centercrop     = torchvision.transforms.CenterCrop((128,128))
centercrop_img = Tensor(np.zeros((128,128,im_slices)))

for i in range(0,im_slices):
    centercrop_img[:,:,i] = centercrop(Tensor(img[:,:,i]))

in_image = np.expand_dims(centercrop_img,0)
in_image = Tensor(in_image).permute(3,0,1,2)


model.eval()
out_trained = model(in_image)
out_image = out_trained["softmax"]

#%% METRICS

model.eval()
out_metrics = model(Tensor(im_flat_test))
seg_metrics = out_metrics["softmax"]

#%% Plot softmax probabilities for a single slice
test_slice = 5
out_img = np.squeeze(seg_metrics[test_slice,:,:,:].detach().numpy())

fig = plt.figure()

class_title = ['Background','Right Ventricle','Myocardium','Left Ventricle']
plt.figure(dpi=200, figsize=(15,15))
for i in range(0,4):
    plt.suptitle('Softmax prob of test image at slice %i' %test_slice, fontsize=20)
    plt.subplot(2, 2, i+1)
    plt.subplots_adjust(hspace = 0.05, wspace = 0)
    plt.imshow(out_img[i,:,:])
    plt.title(class_title[i], fontsize =16)
    plt.xticks(
    rotation=40,
    fontweight='light',
    fontsize=7)
    plt.yticks(
    horizontalalignment='right',
    fontweight='light',
    fontsize=7)
plt.show()   
    
#%% Uncertainty maps (E-maps)   
import scipy.stats

emap = np.zeros((seg_metrics.shape[0],seg_metrics.shape[2],seg_metrics.shape[3]))

#plt.figure(dpi=200)
for i in range(0, emap.shape[0]):
    #plt.suptitle('E-maps', fontsize=16)
    #plt.subplot(3,5,i+1)
    #plt.subplots_adjust(hspace = 0.50, wspace = 0.4)
    out_img = (seg_metrics[i,:,:].detach().numpy())
    entropy2 = scipy.stats.entropy(out_img)
    
    # Normalize 
    m_entropy   = np.max(entropy2)
    entropy     = entropy2/m_entropy
    emap[i,:,:] = entropy

    #plt.imshow(entropy) 
    #plt.xticks(fontsize=0, fontweight='light')
    #plt.yticks(fontsize=0, fontweight='light')
    #plt.title('Slice %i' %i, fontsize =7) 
    #cbar = plt.colorbar(fraction=0.045)
    #cbar.ax.tick_params(labelsize=4) 
    #fig.tight_layout()
    #plt.show

#%% UNCERTAINTY B-MAPS
T = 10 # OBS Set up to find optimal
C = out_image.shape[1]

b = np.zeros((T,out_image.shape[0],out_image.shape[1],out_image.shape[2],out_image.shape[3]))

model.train()
plt.figure(dpi=200)

for i in range(0,T):
    out_trained_bmap = model(in_image)
    out_bmap = out_trained_bmap["softmax"]
    b[i,:,:,:,:] = out_bmap.detach().numpy()
    
    plt.suptitle('Softmax prob. for each of T iterations at slice 4', fontsize=16)
    out_img_bmap = np.squeeze(out_bmap[4,:,:,:].detach().numpy())
    plt.subplot(4,T,i+1)
    plt.subplots_adjust(hspace = 0.0, wspace = 0.1)
    plt.xticks(fontsize=7, fontweight='light')
    plt.yticks(fontsize=7, fontweight='light')
    plt.imshow(out_img_bmap[0,:,:])
    plt.axis('off')
    
    plt.subplot(4,T,i+1+T)
    plt.subplots_adjust(hspace = 0.0, wspace = 0.1)
    plt.xticks(fontsize=7, fontweight='light')
    plt.yticks(fontsize=7, fontweight='light')
    plt.imshow(out_img_bmap[1,:,:])
    plt.axis('off')
    
    plt.subplot(4,T,i+1+T*2)
    plt.subplots_adjust(hspace = 0.0, wspace = 0.1)
    plt.xticks(fontsize=7, fontweight='light')
    plt.yticks(fontsize=7, fontweight='light')
    plt.imshow(out_img_bmap[2,:,:])
    plt.axis('off')
    
    plt.subplot(4,T,i+1+T*3)
    plt.subplots_adjust(hspace = 0.0, wspace = 0.1)
    plt.xticks(fontsize=7, fontweight='light')
    plt.yticks(fontsize=7, fontweight='light')
    plt.imshow(out_img_bmap[3,:,:])
    plt.axis('off')
    
   
pred_mean = (1/T)*sum(b,0)
d = (1/(T-1)) * sum((b-pred_mean)**2,0)

k = np.zeros(b.shape)
for i in range(0,T):
    m = b[i,:,:,:,:]-pred_mean
    k[i,:,:,:,:] = m

b_map     = (1/C) * np.sum(np.sqrt(d),axis=1)


#%% Plot B-maps
#plt.figure(figsize=(10,10))

plt.figure(dpi=200)
for i in range(0, b_map.shape[0]):
    plt.suptitle('B-maps', fontsize =16)
    fig_test = plt.subplot(3,5,i+1)
    plt.subplots_adjust(hspace = 0.50, wspace = 0.35)
    plt.imshow(b_map[i,:,:]) 
    plt.xticks(fontsize=0,fontweight='light')
    plt.yticks(fontsize=0,fontweight='light')
    plt.title('Slice %i' %i, fontsize =7) 
    cbar = plt.colorbar(fraction=0.045)
    cbar.ax.tick_params(labelsize=5)
    #cbar.ax.set_yticklabels(['low','','high'])
    fig.tight_layout()
    plt.show
    

#%% Threshold prediction probabilities

seg_met = np.argmax(seg_metrics.detach().numpy(), axis=1)

# Create Plot 
plt.figure(dpi=200)
plt.suptitle('Comparison of GT and predicted segmentation', fontsize=16 , y=0.8)

#n = 36 # anatomically incoherent 
n = 43 # anatomcally incoherent 
#n = 28 # totally incoherent 

plt.subplot(1, 3, 1)
plt.imshow(np.squeeze(im_flat_test[n,:,:]))
plt.xticks(rotation=40, fontweight='light', fontsize=7,)
plt.yticks(horizontalalignment='right',fontweight='light', fontsize=7,)
plt.title('MRI', fontsize =10)

plt.subplot(1, 3, 2)
plt.imshow(gt_flat_test[n,:,:])
plt.xticks(rotation=40, fontweight='light', fontsize=7,)
plt.yticks(horizontalalignment='right',fontweight='light', fontsize=7,)
plt.title('Ground truth', fontsize =10)

plt.subplot(1, 3, 3)
plt.imshow(seg_met[n,:,:])
plt.xticks(rotation=40, fontweight='light', fontsize=7,)
plt.yticks(horizontalalignment='right',fontweight='light', fontsize=7,)
plt.title('Predicted', fontsize =10)

plt.tight_layout()
plt.show()

#%% calculate metrics
os.chdir("C:/Users/katrine/Documents/GitHub/Speciale2021/")
from metrics import dc, hd, risk, EF_calculation

dice = np.zeros(seg_met.shape[0])
haus = np.zeros(seg_met.shape[0])
fpos = np.zeros(seg_met.shape[0])
fneg = np.zeros(seg_met.shape[0])

for i in range(0,seg_met.shape[0]):
    dice_m  = dc(seg_met[i,:,:],gt_flat_test[i,:,:])  
    dice[i] = dice_m
    
    haus_m  = hd(seg_met[i,:,:],gt_flat_test[i,:,:])  
    haus[i] = haus_m
    
    fn_m, fp_m = risk(seg_met[i,:,:],gt_flat_test[i,:,:])  
    fneg[i] = fn_m
    fpos[i] = fp_m
        

mean_dice = np.mean(dice)  
mean_haus = np.mean(haus)
risk_measure = fpos+fneg


print('mean overall dice = ',mean_dice)  
print('mean overall haus = ',mean_haus)

print('Dice for test slice [n]      = ',dc(seg_met[n,:,:],gt_flat_test[n,:,:]) )
print('Hausdorff for test slice [n] = ',hd(seg_met[n,:,:],gt_flat_test[n,:,:]) )
print('Risk measure test slice [n]  = ', risk_measure[n])

#%% Average Metrics for each structure

seg = torch.nn.functional.one_hot(torch.as_tensor(seg_met), num_classes=4).detach().numpy()
ref = torch.nn.functional.one_hot(Tensor(gt_flat_test).to(torch.int64), num_classes=4).detach().numpy()

dice_c = np.zeros((seg_met.shape[0],3))
haus_c = np.zeros((seg_met.shape[0],3))

# OBS OBS OBS OBS
# dim[0] = BG
# dim[1] = RV
# dim[2] = MYO
# dim[3] = LV

for i in range(0,seg_met.shape[0]):
      
    dice_c[i,0] = dc(seg[i,:,:,1],ref[i,:,:,1])  # = RV
    dice_c[i,1] = dc(seg[i,:,:,2],ref[i,:,:,2])  # = MYO
    dice_c[i,2] = dc(seg[i,:,:,3],ref[i,:,:,3])  # = LV
    
    # If there is no prediction or annotation then don't calculate Hausdorff distance and
    # skip to calculation for next class
    h_count = 0
    
    if len(np.unique(ref[i,:,:,1]))!=1 and len(np.unique(seg[i,:,:,1]))!=1:
        haus_c[i,0]    = hd(seg[i,:,:,1],ref[i,:,:,1])  
    #    print('haus_rv  for i = ',i)
        h_count += 1
    else:
        pass
    
    if len(np.unique(ref[i,:,:,2]))!=1 and len(np.unique(seg[i,:,:,2]))!=1:      
        haus_c[i,1]    = hd(seg[i,:,:,2],ref[i,:,:,2])  
    #    print('haus_myo for i = ',i)     
        h_count += 1
    else:
        pass
    
    if len(np.unique(ref[i,:,:,3]))!=1 and len(np.unique(seg[i,:,:,3]))!=1:
        haus_c[i,2]    = hd(seg[i,:,:,3],ref[i,:,:,3])  
    #    print('haus_lv  for i = ',i)
        h_count += 1
    else:
        pass
    
        pass        
    if h_count!= 3:
        print('Haus not calculated for all classes for slice: ', i)
    else:
        pass 
    
mean_dice = np.mean(dice_c, axis=0)  
mean_haus = np.mean(haus_c, axis=0)
print('mean dice = ',mean_dice)  
print('mean haus = ',mean_haus)


#%% Coverage threshold
from uncertain_thres import get_seg_errors_mask, generate_thresholds

pred = Tensor(seg).permute(0,3,1,2)

err_indices = get_seg_errors_mask(pred, gt_flat_test)
percentiles = generate_thresholds(pred, gt_flat_test, emap)

uncertain_voxels = np.zeros((len(percentiles),48,H,W))
cov_slices = np.zeros((seg.shape[0],len(percentiles)))

for p, thres in enumerate(percentiles):
    for i in range(0,seg.shape[0]):
        uncertain_voxels[p,i,:,:] = emap[i,:,:] >= thres
        cov_slices[i,p] = np.sum(uncertain_voxels[p,i,:,:])
    coverage = np.mean(cov_slices*1/(H*W)*100, axis=0)

#%% Risk measure

#thresholding the softmax for each of 4 channels by the percentile values
prob_thres = np.zeros((len(percentiles),48,4,H,W))
a = np.zeros((48,4,H,W))
aa = np.zeros((len(percentiles),48,H,W))
risk_measure = np.zeros((len(percentiles),48))

for i in range(len(percentiles)):
    for k in range(0,prob_thres.shape[1]):
        a[k,:,:,:]  = seg_metrics[k,:,:,:] > percentiles[i]
        aa[i,:,:,:] = np.argmax(a,axis=1)
        
        fn_m, fp_m  = risk(aa[i,k,:,:],gt_flat_test[k,:,:])  
        
        risk_measure[i,k] = fp_m + fn_m
risk_m = np.mean(risk_measure, axis=1)

#%%
plt.figure(dpi=300)

thres_values = np.round([percentiles[50],percentiles[70], percentiles[80], percentiles[90]],8)
test_slice = 0

plt.suptitle('Comparison of uncertainty threholds', y=0.9)

plt.subplot(2,4,1)
plt.imshow(im_flat_test[test_slice,0,:,:])
plt.subplots_adjust(hspace = 0, wspace = 0.6)
plt.title('Original Im', fontsize=10)

plt.subplot(2,4,2)
plt.imshow(gt_flat_test[test_slice,:,:])
plt.title('GT seg.', fontsize=10)

plt.subplot(2,4,3)
plt.imshow(seg_met[test_slice,:,:])
plt.title('Predicted seg.', fontsize=10)

plt.subplot(2,4,4)
plt.imshow(emap[test_slice,:,:])
plt.title('E-map', fontsize=10)
plt.colorbar(fraction=0.05)

plt.subplot(2,4,5)
plt.imshow(aa[50,test_slice,:,:])
plt.title('Thres: 8.27 e-12', fontsize=10)

plt.subplot(2,4,6)
plt.imshow(aa[70,test_slice,:,:])
plt.title('Thres: 7.40 e-07', fontsize=10)

plt.subplot(2,4,7)
plt.imshow(aa[80,test_slice,:,:])
plt.title('Thres: 1.90 e-04', fontsize=10)

plt.subplot(2,4,8)
plt.imshow(aa[90,test_slice,:,:])
plt.title('Thres: 3.55 e-02', fontsize=10)

#%%
# Risk-coverage curve
plt.figure(dpi=300)
plt.suptitle('Risk-Coverage curve', fontsize=16)
plt.plot(coverage[1:-1],risk_m[1:-1],'b.', label ='Unet-CE (e-map)')
plt.xlabel('Coverage [%]')
plt.ylabel('Risk (FP+FN)')


