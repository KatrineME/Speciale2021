#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

# recursive implementation of Unet
import torch

from torch import nn


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
        # construct unet structure
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

#%% Load packages
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
from torch import Tensor

#!pip install opencv-python

#%% Load image  
cwd = os.getcwd()
os.chdir("C:/Users/katrine/Documents/Universitet/Speciale/ACDC_training_data/training")
#os.chdir('/Users/michalablicher/Desktop/training')


frame_dia_im = np.sort(glob2.glob('patient*/**/patient*_frame01.nii.gz'))

num_patients = 4#len(frame_dia_im)
H = 128
W = 128
in_c = 1
slices = 11

data_im = []#np.zeros((num_patients,slices,in_c,H,W))
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
    #print('in_image shape',in_image.shape)
    
    #data_im[i] = in_image
    data_im.append(in_image.astype(object))
    #print('Patient done', i)

#%% Load annotations
frame_dia_gt = np.sort(glob2.glob('patient*/**/patient*_frame01_gt.nii.gz'))

num_patients = 4#len(frame_dia_gt)
H = 128
W = 128
slices = 10

data_gt = [] #np.zeros((num_patients,slices,H,W))
centercrop     = torchvision.transforms.CenterCrop((H,W))

for i in range(0,num_patients):
    n_gt = nib.load(frame_dia_gt[i])
    gt  = n_gt.get_fdata()
    
    gt_slices      = gt.shape[2]
    centercrop_gt = Tensor(np.zeros((H,W,gt_slices)))
    
    for j in range(0,gt_slices):
        centercrop_gt[:,:,j] = centercrop(Tensor(gt[:,:,j]))
   
    in_gt = Tensor(centercrop_gt).permute(2,0,1).detach().numpy()
    #print('in_gt shape',in_gt.shape)
    
    #data_gt[i] = in_gt
    data_gt.append(in_gt.astype(object))
    #print('Patient done', i)

#%% Setting up training loop
import torch.optim as optim
from torch.autograd import Variable

LEARNING_RATE = 0.001 
criterion    = nn.CrossEntropyLoss() 
#criterion     = nn.BCELoss()
#criterion     = SoftDice
#criterion     = Brier

# weight_decay is equal to L2 regularizationst
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# and a learning rate scheduler which decreases the learning rate by 10x every 3 epochs
#lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
#                                               step_size=3,
#                                               gamma=0.1)

num_epoch = 4

#%% BATCH GENERATOR

num_train = 2 #1

im_flat_train = np.concatenate(data_im[0:num_train]).astype(None)
gt_flat_train = np.concatenate(data_gt[0:num_train]).astype(None)

im_flat_eval = np.concatenate(data_im[num_train:len(data_im)]).astype(None)
gt_flat_eval = np.concatenate(data_gt[num_train:len(data_gt)]).astype(None)


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
        print('i=',i)

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
        losses.append(train_loss)
        (epoch + 1, i + 1, train_loss)
        train_loss = 0.0
     
    model.eval()
    print('Epoch eval=',epoch)
    eval_loss = 0.0  
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        #inputs, labels = data
        inputs = Tensor(im_flat_eval)
        labels = Tensor(gt_flat_eval)
        print('i=',i)

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
        losses_eval.append(eval_loss)
        (epoch + 1, i + 1, eval_loss)
        eval_loss = 0.0

print('Finished Training')
        
#%%    
epochs = np.arange(len(losses))
epochs_eval = np.arange(len(losses_eval))
plt.figure()
plt.plot(epochs, losses, 'b', label='Training Loss')
plt.plot(epochs_eval, losses_eval, 'r', label='Validation Loss')
plt.legend(loc="upper right")
plt.title("Loss function")
plt.show()
print('Finished Training')


#%% TESTING! 
# Load new image for testing
cwd = os.getcwd()
os.chdir("C:/Users/katrine/Documents/Universitet/Speciale/ACDC_training_data/training/patient008")
#os.chdir('/Users/michalablicher/Desktop/training/patient008')

nimg = nib.load('patient008_frame01.nii.gz')
img  = nimg.get_data()

im_slices = img.shape[2]

for i in range(0,im_slices):
    plt.suptitle('Test image slices')
    plt.subplot(2,5,i+1)
    plt.imshow(img[:,:,i])    
    
#%%
# Crop image
centercrop     = torchvision.transforms.CenterCrop((128,128))
centercrop_img = Tensor(np.zeros((128,128,im_slices)))

for i in range(0,im_slices):
    centercrop_img[:,:,i] = centercrop(Tensor(img[:,:,i]))
    #plt.subplot(3,4,i+1)
    #plt.imshow(centercrop_img[:,:,i])

in_image = np.expand_dims(centercrop_img,0)
in_image = Tensor(in_image).permute(3,0,1,2)

out_trained = model(in_image)
out_image = out_trained["softmax"]
#%%

test_slice = 4
out_img = np.squeeze(out_image[test_slice,:,:,:].detach().numpy())

fig = plt.figure()
plt.figure(figsize=(15,15))
   
class_title = ['Background','Right Ventricle','Myocardium','Left Ventricle']
for i in range(0,4):

    plt.suptitle('Softmax prob of test image', fontsize=40)
    plt.subplot(2, 2, i+1)
    plt.subplots_adjust(hspace = 0.15, wspace = 0.25)
    plt.imshow(out_img[i,:,:])
    #plt.xlabel('Property type')
    #plt.ylabel('Number of listings')
    #plt.xticks(rotation=90)
    plt.title(class_title[i], fontsize =16)
    plt.xticks(
    rotation=40,
    #horizontalalignment='right',
    fontweight='light',
    fontsize=10,
    )
    plt.yticks(
    #rotation=40,
    horizontalalignment='right',
    fontweight='light',
    fontsize=10,
    )
plt.show()   
save_image(img1, 'img1.png')
    
#%% Uncertainty maps (E-maps)   
import scipy.stats
#plt.figure(figsize=(15,15))
for i in range(0, out_image.shape[0]):
    plt.suptitle('E-maps')
    plt.subplot(3,5,i+1)
    plt.subplots_adjust(hspace = 0.25, wspace = 0.4)
    out_img = (out_image[i,:,:].detach().numpy())
    entropy2 = scipy.stats.entropy(out_img)
    plt.imshow(entropy2) 
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    plt.title('Slice %i' %i, fontsize =12) 
    

#%% UNCERTAINTY B-MAPS
T = 10 # Find optimal
C = out_image.shape[1]
#smax_prob = np.array([[10],[10],[4],[128,128]], np.int32)

b = np.zeros((T,out_image.shape[0],out_image.shape[1],out_image.shape[2],out_image.shape[3]))

for i in range(0,T):
    out_trained_bmap = model(in_image)
    out_bmap = out_trained_bmap["softmax"]
    b[i,:,:,:,:] = out_bmap.detach().numpy()
    
    plt.suptitle('Softmax for each of T iterations at slice 4')
    out_img_bmap = np.squeeze(out_bmap[4,:,:,:].detach().numpy())
    plt.subplot(4,T,i+1)
    plt.subplots_adjust(hspace = 0.30, wspace = 0.50)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.imshow(out_img_bmap[0,:,:])
    plt.subplot(4,T,i+1+T)
    plt.subplots_adjust(hspace = 0.30, wspace = 0.50)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.imshow(out_img_bmap[1,:,:])
    plt.subplot(4,T,i+1+T*2)
    plt.subplots_adjust(hspace = 0.30, wspace = 0.50)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.imshow(out_img_bmap[2,:,:])
    plt.subplot(4,T,i+1+T*3)
    plt.subplots_adjust(hspace = 0.30, wspace = 0.50)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.imshow(out_img_bmap[3,:,:])

pred_mean = (1/T)*sum(b,0)
d = (1/(T-1)) * sum((b-pred_mean)**2,0)

b_map     = (1/C) * np.sum(np.sqrt(d),axis=1) 

#%%
#plt.figure(figsize=(10,10))

for i in range(0, b_map.shape[0]):
    plt.suptitle('B-maps', fontsize =16)
    fig_test = plt.subplot(2,5,i+1)
    plt.subplots_adjust(bottom=0.3, top=0.9, wspace=0.35)#hspace = 0.01, wspace = 0.4)
    plt.imshow(b_map[i,:,:]) 
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    plt.title('Slice %i' %i, fontsize =11) 
    fig.tight_layout()
    plt.show

#%%
os.chdir("C:/Users/katrine/Documents/Universitet/Speciale")

from test_script import normal_func 


m,s = normal_func(out_image[1,1,:,:])

#%%
plt.imshow(out_image[1,1,:,:].detach().numpy())
m = np.mean(out_image[1,1,:,:],axis=None, dtype=None)
