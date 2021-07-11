#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 15:29:54 2021

@author: michalablicher
"""

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
import scipy.ndimage
import scipy.stats
import torchsummary


if torch.cuda.is_available():
    # Tensor = torch.cuda.FloatTensor
    device = 'cuda'
else:
    # Tensor = torch.FloatTensor
    device = 'cpu'
torch.cuda.manual_seed_all(808)

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
    unet = BayesUNet(num_classes=4, in_channels=1, drop_prob=0.1)
    
    if device == 'cuda':
        unet.cuda()
    #torchsummary.summary(unet, (1, 128, 128))

#%% Specify directory
if device == 'cuda':
    user = 'GPU'
else:
    user = 'K'
#%%
user = 'M'
if user == 'M':
    os.chdir('/Users/michalablicher/Documents/GitHub/Speciale2021')
if user == 'K':
    os.chdir('C:/Users/katrine/Documents/GitHub/Speciale2021')
if user == 'GPU':
    os.chdir('/home/michala/Speciale2021/Speciale2021')

 
from load_data_gt_im_sub_space import load_data_sub

phase = 'Systole'

data_im_es_DCM,  data_gt_es_DCM  = load_data_sub(user,phase,'DCM')
data_im_es_HCM,  data_gt_es_HCM  = load_data_sub(user,phase,'HCM')
data_im_es_MINF, data_gt_es_MINF = load_data_sub(user,phase,'MINF')
data_im_es_NOR,  data_gt_es_NOR  = load_data_sub(user,phase,'NOR')
data_im_es_RV,   data_gt_es_RV   = load_data_sub(user,phase,'RV')

phase = 'Diastole'

data_im_ed_DCM,  data_gt_ed_DCM  = load_data_sub(user,phase,'DCM')
data_im_ed_HCM,  data_gt_ed_HCM  = load_data_sub(user,phase,'HCM')
data_im_ed_MINF, data_gt_ed_MINF = load_data_sub(user,phase,'MINF')
data_im_ed_NOR,  data_gt_ed_NOR  = load_data_sub(user,phase,'NOR')
data_im_ed_RV,   data_gt_ed_RV   = load_data_sub(user,phase,'RV')


#%% BATCH GENERATOR
num_train_sub = 12
num_eval_sub = num_train_sub
num_test_sub = num_eval_sub + 8
"""
im_train_ed_sub = np.concatenate((np.concatenate(data_im_ed_DCM[0:num_train_sub]).astype(None),
                                  np.concatenate(data_im_ed_HCM[0:num_train_sub]).astype(None),
                                  np.concatenate(data_im_ed_MINF[0:num_train_sub]).astype(None),
                                  np.concatenate(data_im_ed_NOR[0:num_train_sub]).astype(None),
                                  np.concatenate(data_im_ed_RV[0:num_train_sub]).astype(None)))

gt_train_ed_sub = np.concatenate((np.concatenate(data_gt_ed_DCM[0:num_train_sub]).astype(None),
                                  np.concatenate(data_gt_ed_HCM[0:num_train_sub]).astype(None),
                                  np.concatenate(data_gt_ed_MINF[0:num_train_sub]).astype(None),
                                  np.concatenate(data_gt_ed_NOR[0:num_train_sub]).astype(None),
                                  np.concatenate(data_gt_ed_RV[0:num_train_sub]).astype(None)))

gt_test_ed_sub = gt_train_ed_sub
im_test_ed_sub = im_train_ed_sub
"""
im_test_ed_sub = np.concatenate((np.concatenate(data_im_ed_DCM[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_im_ed_HCM[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_im_ed_MINF[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_im_ed_NOR[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_im_ed_RV[num_eval_sub:num_test_sub]).astype(None)))

gt_test_ed_sub = np.concatenate((np.concatenate(data_gt_ed_DCM[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_gt_ed_HCM[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_gt_ed_MINF[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_gt_ed_NOR[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_gt_ed_RV[num_eval_sub:num_test_sub]).astype(None)))

im_test_es_sub = np.concatenate((np.concatenate(data_im_es_DCM[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_im_es_HCM[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_im_es_MINF[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_im_es_NOR[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_im_es_RV[num_eval_sub:num_test_sub]).astype(None)))

gt_test_es_sub = np.concatenate((np.concatenate(data_gt_es_DCM[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_gt_es_HCM[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_gt_es_MINF[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_gt_es_NOR[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_gt_es_RV[num_eval_sub:num_test_sub]).astype(None)))

print('Data loaded+concat')


#%% Load model if averagered on GPU

path_out_soft = '/Users/michalablicher/Desktop/Out_softmax_fold_avg_150dia_CE.pt'
#path_out_soft = 'C:/Users/katrine/Desktop/Optuna/Final CV models/Out_softmax_fold_avg_100sys_CE.pt'

out_soft = torch.load(path_out_soft ,  map_location=torch.device(device))

#%%
#Plot softmax probabilities for a single slice
test_slice = 199
alpha = 0.4

# Slices 9, 28, 67, 84, 177, 186, 199, 248, 255, 265, 269, 312, 315

fig = plt.figure()

class_title = ['Background','Right Ventricle','Myocardium','Left Ventricle']
ref_dia = torch.nn.functional.one_hot(Tensor(gt_test_es_sub).to(torch.int64), num_classes=4)
plt.figure(dpi=200, figsize=(18,32))

w = 0.1

for fold_model in range (0,6):
    out_img_ed = np.squeeze(out_soft[fold_model,test_slice,:,:,:])
    seg_met_dia = np.argmax(out_soft[fold_model,:,:,:], axis=1)
    seg_dia = torch.nn.functional.one_hot(torch.as_tensor(seg_met_dia), num_classes=4)
    
    #Reference annotation
    plt.suptitle('Diastolic phase: test image at slice %i for CV folds' %test_slice, fontsize=30, y=0.9)
    plt.subplot(7, 4, 1)
    plt.subplots_adjust(hspace = 0.05, wspace = w)
    plt.imshow(ref_dia[test_slice,:,:,0])
    plt.imshow(im_test_es_sub[test_slice,0,:,:],alpha=alpha)
    plt.ylabel('Reference', fontsize=16)
    plt.title('Background', fontsize=16)
    
    plt.subplot(7, 4, 2)
    plt.subplots_adjust(hspace = 0.05, wspace = w)
    plt.imshow(ref_dia[test_slice,:,:,1])
    plt.imshow(im_test_es_sub[test_slice,0,:,:],alpha=alpha)
    plt.title('Right ventricle', fontsize=16)
    
    plt.subplot(7, 4, 3)
    plt.subplots_adjust(hspace = 0.05, wspace = w)
    plt.imshow(ref_dia[test_slice,:,:,2])
    plt.imshow(im_test_es_sub[test_slice,0,:,:],alpha=alpha)
    plt.title('Myocardium', fontsize=16)
    
    plt.subplot(7, 4, 4)
    plt.subplots_adjust(hspace = 0.05, wspace = w)
    plt.imshow(ref_dia[test_slice,:,:,3])
    plt.imshow(im_test_es_sub[test_slice,0,:,:],alpha=alpha)
    plt.title('Left ventricle', fontsize=16)
    
    
    #CV model segmentations
    plt.subplot(7, 4, 1+4*(fold_model+1))
    plt.subplots_adjust(hspace = 0.05, wspace = w)
    plt.imshow(out_soft[fold_model, test_slice,0,:,:])
    plt.imshow(im_test_es_sub[test_slice,0,:,:],alpha=alpha)
    plt.ylabel('CV fold {}'.format(fold_model), fontsize=16)
    
    plt.subplot(7, 4, 2+4*(fold_model+1))
    plt.subplots_adjust(hspace = 0.05, wspace = w)
    plt.imshow(out_soft[fold_model, test_slice,1,:,:])
    plt.imshow(im_test_es_sub[test_slice,0,:,:],alpha=alpha)
    
    plt.subplot(7, 4, 3+4*(fold_model+1))
    plt.subplots_adjust(hspace = 0.05, wspace = w)
    plt.imshow(out_soft[fold_model, test_slice,2,:,:])
    plt.imshow(im_test_es_sub[test_slice,0,:,:],alpha=alpha)
    
    plt.subplot(7, 4, 4+4*(fold_model+1))
    plt.subplots_adjust(hspace = 0.05, wspace = w)
    plt.imshow(out_soft[fold_model, test_slice,3,:,:])
    plt.imshow(im_test_es_sub[test_slice,0,:,:],alpha=alpha)

plt.show()  


#%%
#Plot softmax probabilities for a single slice
test_slice = 45
alpha = 0.4

# Slices 9, 28, 67, 84, 177, 186, 199, 248, 255, 265, 269, 312, 315

fig = plt.figure()

class_title = ['Background','Right Ventricle','Myocardium','Left Ventricle']
ref_dia = torch.nn.functional.one_hot(Tensor(gt_test_ed_sub).to(torch.int64), num_classes=4)
plt.figure(dpi=200, figsize=(18,32))


back_image = im_test_ed_sub[test_slice,0,:,:]

w = 0.1

for fold_model in range (0,6):
    out_img_ed = np.squeeze(out_soft[fold_model,test_slice,:,:,:])
    seg_met_dia = np.argmax(out_soft[fold_model,:,:,:], axis=1)
    seg_dia = torch.nn.functional.one_hot(torch.as_tensor(seg_met_dia), num_classes=4)
    
    #Reference annotation
    plt.suptitle('Diastolic phase: test image at slice %i for CV folds' %test_slice, fontsize=30, y=0.9)
    plt.subplot(7, 4, 1)
    plt.subplots_adjust(hspace = 0.05, wspace = w)
    plt.imshow(ref_dia[test_slice,:,:,0])
    plt.imshow(back_image,alpha=alpha)
    plt.ylabel('Reference', fontsize=16)
    plt.title('Background', fontsize=16)
    
    plt.subplot(7, 4, 2)
    plt.subplots_adjust(hspace = 0.05, wspace = w)
    plt.imshow(ref_dia[test_slice,:,:,1])
    plt.imshow(back_image,alpha=alpha)
    plt.title('Right ventricle', fontsize=16)
    
    plt.subplot(7, 4, 3)
    plt.subplots_adjust(hspace = 0.05, wspace = w)
    plt.imshow(ref_dia[test_slice,:,:,2])
    plt.imshow(back_image,alpha=alpha)
    plt.title('Myocardium', fontsize=16)
    
    plt.subplot(7, 4, 4)
    plt.subplots_adjust(hspace = 0.05, wspace = w)
    plt.imshow(ref_dia[test_slice,:,:,3])
    plt.imshow(back_image,alpha=alpha)
    plt.title('Left ventricle', fontsize=16)
    
    
    #CV model segmentations
    plt.subplot(7, 4, 1+4*(fold_model+1))
    plt.subplots_adjust(hspace = 0.05, wspace = w)
    plt.imshow(seg_dia[test_slice,:,:,0])
    plt.imshow(back_image,alpha=alpha)
    plt.ylabel('CV fold {}'.format(fold_model), fontsize=16)
    
    plt.subplot(7, 4, 2+4*(fold_model+1))
    plt.subplots_adjust(hspace = 0.05, wspace = w)
    plt.imshow(seg_dia[test_slice,:,:,1])
    plt.imshow(back_image,alpha=alpha)
    
    plt.subplot(7, 4, 3+4*(fold_model+1))
    plt.subplots_adjust(hspace = 0.05, wspace = w)
    plt.imshow(seg_dia[test_slice,:,:,2])
    plt.imshow(back_image,alpha=alpha)
    
    plt.subplot(7, 4, 4+4*(fold_model+1))
    plt.subplots_adjust(hspace = 0.05, wspace = w)
    plt.imshow(seg_dia[test_slice,:,:,3])
    plt.imshow(back_image,alpha=alpha)

plt.show()  

#%% Mean + argmax + one hot

out_soft_mean   = out_soft.mean(axis=0)

#out_soft_mean   = out_soft[5,:,:,:,:]
out_seg_mean_am = np.argmax(out_soft_mean, axis=1)
out_seg_mean    = torch.nn.functional.one_hot(torch.as_tensor(out_seg_mean_am), num_classes=4).detach().cpu().numpy()

ref = torch.nn.functional.one_hot(torch.as_tensor(Tensor(gt_test_ed_sub).to(torch.int64)), num_classes=4).detach().cpu().numpy()

#%%
w = 0.1
h = 0.3
test_slice = 300
plt.figure(dpi=200)
plt.suptitle('Diastolic - Averaged model for test image at slice: {}'.format(test_slice))

plt.subplot(2,2,1)
plt.subplots_adjust(hspace = h, wspace = w)
plt.imshow(out_soft_mean[test_slice,0,:,:])
#plt.imshow(im_test_ed_sub[test_slice,0,:,:],alpha=alpha)
plt.title('Background', fontsize=10)

plt.subplot(2,2,2)
plt.subplots_adjust(hspace = h, wspace = w)
plt.imshow(out_soft_mean[test_slice,1,:,:])
plt.title('Right ventricle', fontsize=10)

plt.subplot(2,2,3)
plt.subplots_adjust(hspace = h, wspace = w)
plt.imshow(out_soft_mean[test_slice,2,:,:])
plt.title('Myocardium', fontsize=10)

plt.subplot(2,2,4)
plt.subplots_adjust(hspace = h, wspace = w)
plt.imshow(out_soft_mean[test_slice,3,:,:])
plt.title('Left ventricle', fontsize=10)

#%%
test_slice = 25*0
plt.figure(figsize=(15,15),dpi=200)
for i in range(0,25):
    plt.subplot(5,5,i+1)
    plt.imshow(im_test_es_sub[i + test_slice,0,:,:])
    #plt.imshow(ref[i + test_slice,:,:,1])
    plt.title('slice: {}'.format(i + test_slice), fontsize =15)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)


#ref = torch.nn.functional.one_hot(torch.as_tensor(Tensor(gt_test_ed_sub).to(torch.int64)), num_classes=4).detach().cpu().numpy()

#%%
plt.figure(dpi=200)
plt.subplot(1,2,1)
plt.imshow(im_test_es_sub[187,0,:,:])
plt.title('Original MRI')
plt.subplot(1,2,2)
plt.imshow(im_test_es_sub[187,0,:,:])
plt.imshow(gt_test_es_sub[187,:,:], alpha=0.6)
plt.title('Reference without RV')


#%% 
plt.figure(figsize=(15,15),dpi=200)
test_slice = 31
plt.imshow(out_seg_mean_am[test_slice,:,:])
plt.title('slice: {}'.format(test_slice), fontsize =25)


#%%
def lv_loss(y_pred):
    Y_BGR  = y_pred[:,0,:,:]           # size([B,H,W])
    Y_RV   = y_pred[:,1,:,:]           # size([B,H,W])
    Y_LV   = y_pred[:,3,:,:]           # size([B,H,W])

    Y_LV_pad = torch.nn.functional.pad(Y_LV,(1,1,1,1),'constant', 0)

    Y_up   = Y_LV_pad[:,2:130,1:129]
    Y_down = Y_LV_pad[:,0:128,1:129]
    
    Y_left = Y_LV_pad[:,1:129,2:130]
    Y_right= Y_LV_pad[:,1:129,0:128]
    

    inside = (Y_up + Y_down + Y_left + Y_right) * (Y_BGR + Y_RV)
    inside = inside.detach().cpu()#cuda()

    #print('inside', inside)    
    return inside # torch.sum(Tensor(inside))/(128*128*32)#.cuda()

out_seg_per = Tensor(out_seg_mean).permute(0,3,1,2)
lv_neigh = lv_loss(out_seg_per)

c_non = np.count_nonzero(lv_neigh, axis = (1,2)) # number of error pixels in each slice

cnon_slice = np.count_nonzero(c_non) # number of slices with erros 
print('Number of slices with errors:', cnon_slice)
print('Percentage of slices with errors:', (cnon_slice/len(c_non))*100,'%')
print('Number of errornous neighbour pixels:', c_non.sum())


# Slices per patient
p = []

for i in range(0,8):
    p.append(data_gt_ed_DCM[num_eval_sub:num_test_sub][i].shape[0])
for i in range(0,8):
    p.append(data_gt_ed_HCM[num_eval_sub:num_test_sub][i].shape[0])
for i in range(0,8):
    p.append(data_gt_ed_MINF[num_eval_sub:num_test_sub][i].shape[0])
for i in range(0,8):
    p.append(data_gt_ed_NOR[num_eval_sub:num_test_sub][i].shape[0])
for i in range(0,8):
    p.append(data_gt_ed_RV[num_eval_sub:num_test_sub][i].shape[0])
    
test_index = len(p)

s = 0
cnon_pt = np.zeros(test_index)

for i in range(0,test_index):
    for j in range(0, p[i]):
        cnon_pt[i] += np.count_nonzero(c_non[j+s])
        
    s += p[i] 
    #print('s= ',s)
print('Number of patient volumes w. errors:',np.count_nonzero(cnon_pt))
print('Percentage of patient volumes w. errors:',(np.count_nonzero(cnon_pt)/len(p))*100)   

#%%
import skimage
from skimage import measure

labeled_image_rv = []
labeled_image_myo = []
labeled_image_lv = []
labeled_image_all = []

out_seg_mean_bin = (out_seg_mean_am > 0).astype(int)

#w = out_seg_mean_bin.astype(int)

#%%
for i in range(0, (out_seg_mean.shape[0])):
    labeled_image_all.append(skimage.measure.label(out_seg_mean_bin[i,:,:], connectivity=2, return_num=True))
    labeled_image_rv.append(skimage.measure.label(out_seg_mean[i,:,:,1], connectivity=2, return_num=True))
    labeled_image_myo.append(skimage.measure.label(out_seg_mean[i,:,:,2], connectivity=2, return_num=True))
    labeled_image_lv.append(skimage.measure.label(out_seg_mean[i,:,:,3], connectivity=2, return_num=True))

#%%

slice = 79
plt.figure(dpi=200)
plt.subplot(2,2,1)
plt.imshow(out_seg_mean_am[slice,:,:])
plt.subplot(2,2,2)
plt.imshow(labeled_image_rv[slice][0])
plt.subplot(2,2,3)
plt.imshow(labeled_image_myo[slice][0])
plt.subplot(2,2,4)
plt.imshow(labeled_image_all[slice][0])

#%%
multi_lab_rv = []
multi_lab_myo = []
multi_lab_lv = []
multi_lab_all = []

for i in range(0, (out_seg_mean.shape[0])):
    multi_lab_rv.append(float(labeled_image_rv[i][1] > 1))
    multi_lab_myo.append(float(labeled_image_myo[i][1] > 1))
    multi_lab_lv.append(float(labeled_image_lv[i][1] > 1))
    multi_lab_all.append(float(labeled_image_all[i][1] > 1))

    
tot_all = np.sum(multi_lab_all)

tot_rv = np.sum(multi_lab_rv)
tot_myo = np.sum(multi_lab_myo)
tot_lv = np.sum(multi_lab_lv)

print('Total slices with more:', tot_rv, tot_myo, tot_lv, tot_all)
#%%
gt_per = (Tensor(ref).permute(0,3,1,2)).detach().numpy()
out_seg_per = (Tensor(out_seg_mean).permute(0,3,1,2)).detach().numpy()

ss = np.sum(out_seg_per, axis=(2,3))
gs = np.sum(gt_per, axis=(2,3))

emp_both = (ss == 0) & (gs == 0)
bin_both = emp_both.sum(axis=0)

g_emp = gs == 0
bin_gt = g_emp.sum(axis=0)

seg_emp = ss == 0
bin_seg = seg_emp.sum(axis=0)

print('bin_both', bin_both)
print('bin_seg', bin_seg)


#%%
w = 0.1
h = 0.3
test_slice = 165
plt.figure(dpi=200)
plt.suptitle('Diastolic - Averaged model for test image at slice: {}'.format(test_slice))

plt.subplot(2,2,1)
plt.subplots_adjust(hspace = h, wspace = w)
plt.imshow(out_seg_mean[test_slice,:,:,0])
plt.imshow(im_test_ed_sub[test_slice,0,:,:],alpha=alpha)
plt.title('Background', fontsize=10)

plt.subplot(2,2,2)
plt.subplots_adjust(hspace = h, wspace = w)
plt.imshow(out_seg_mean[test_slice,:,:,1])
plt.imshow(im_test_ed_sub[test_slice,0,:,:],alpha=alpha)
plt.title('Right ventricle', fontsize=10)

plt.subplot(2,2,3)
plt.subplots_adjust(hspace = h, wspace = w)
plt.imshow(out_seg_mean[test_slice,:,:,2])
plt.imshow(im_test_ed_sub[test_slice,0,:,:],alpha=alpha)
plt.title('Myocardium', fontsize=10)

plt.subplot(2,2,4)
plt.subplots_adjust(hspace = h, wspace = w)
plt.imshow(out_seg_mean[test_slice,:,:,3])
plt.imshow(im_test_ed_sub[test_slice,0,:,:],alpha=alpha)
plt.title('Left ventricle', fontsize=10)

#%% Metrics
#os.chdir("C:/Users/katrine/Documents/GitHub/Speciale2021")
os.chdir('/Users/michalablicher/Documents/GitHub/Speciale2021')
from metrics import accuracy_self, EF_calculation, dc, hd, jc, precision, mcc, recall, risk, sensitivity, specificity, true_negative_rate, true_positive_rate, positive_predictive_value, hd95, assd, asd, ravd, volume_correlation, volume_change_correlation, obj_assd, obj_asd, obj_fpr, obj_tpr

dice = np.zeros((out_seg_mean.shape[0],4))
haus = np.zeros((out_seg_mean.shape[0],3))
haus95 = np.zeros((out_seg_mean.shape[0],3))

# OBS OBS OBS OBS
# dim[0] = BG
# dim[1] = RV
# dim[2] = MYO
# dim[3] = LV

for i in range(0,out_seg_mean.shape[0]):
      
    dice[i,0] = dc(out_seg_mean[i,:,:,1],ref[i,:,:,1])  # = RV
    dice[i,1] = dc(out_seg_mean[i,:,:,2],ref[i,:,:,2])  # = MYO
    dice[i,2] = dc(out_seg_mean[i,:,:,3],ref[i,:,:,3])  # = LV
    dice[i,3] = dc(out_seg_mean[i,:,:,0],ref[i,:,:,0])
    
    # If there is no prediction or annotation then don't calculate Hausdorff distance and
    # skip to calculation for next class
    h_count = 0
    
    if len(np.unique(ref[i,:,:,1]))!=1 and len(np.unique(out_seg_mean[i,:,:,1]))!=1:
        haus[i,0]    = hd(out_seg_mean[i,:,:,1],ref[i,:,:,1])  
        haus95[i,0]    = hd95(out_seg_mean[i,:,:,1],ref[i,:,:,1])  
        h_count += 1
    else:
        pass
    
    if len(np.unique(ref[i,:,:,2]))!=1 and len(np.unique(out_seg_mean[i,:,:,2]))!=1:      
        haus[i,1]    = hd(out_seg_mean[i,:,:,2],ref[i,:,:,2]) 
        haus95[i,1]    = hd95(out_seg_mean[i,:,:,2],ref[i,:,:,2])
        h_count += 1
    else:
        pass
    
    if len(np.unique(ref[i,:,:,3]))!=1 and len(np.unique(out_seg_mean[i,:,:,3]))!=1:
        haus[i,2]    = hd(out_seg_mean[i,:,:,3],ref[i,:,:,3])
        haus95[i,2]    = hd95(out_seg_mean[i,:,:,3],ref[i,:,:,3])
        h_count += 1
    else:
        pass
    
        pass        
    if h_count!= 3:
        print('Haus not calculated for all classes for slice: ', i)
    else:
        pass 
    
mean_dice = np.mean(dice, axis=0)  
std_dice  = np.std(dice,  axis=0)
var_dice  = np.var(dice,  axis=0)

mean_haus = np.mean(haus, axis=0)
std_haus  = np.std(haus,  axis=0)
var_haus  = np.var(haus,  axis=0)

mean_haus95 = np.mean(haus95, axis=0)
std_haus95  = np.std(haus95,  axis=0)
var_haus95  = np.var(haus95,  axis=0)

print('mean dice   = ',mean_dice)  
print('std dice    = ',  std_dice) 
print('var dice    = ',  var_dice) 

print('mean haus   = ',mean_haus)
print('mean haus95 = ',mean_haus95)

print('std haus    = ',  std_haus)
print('std haus95  = ',  std_haus95)

print('var haus    = ', var_haus) 
print('var haus95  = ', var_haus95)

#%% Boxplot
plt.figure(dpi=200, figsize =(12,12))
plt.subplot(2,1,1)
plt.hist(dice, label=('RV','MYO','LV'), color=('tab:blue','mediumseagreen','orange'))
plt.legend()
plt.xlabel('Dice Score')
plt.ylabel('# Number of slices')
plt.title('Histogram of Dice Scores for CE model')
plt.grid(True, color = "grey", linewidth = "0.5", linestyle = "-")

plt.subplot(2,1,2)
plt.hist(dice, label=('RV','MYO','LV'), color=('tab:blue','mediumseagreen','orange'))
plt.legend()
plt.xlabel('Dice Score')
plt.ylabel('# Number of slices')
plt.title('Histogram of Dice Scores for SD model')
plt.grid(True, color = "grey", linewidth = "0.5", linestyle = "-")


#%% ACCURACY
acc = np.zeros((out_seg_mean.shape[0],3))

for i in range(0,out_seg_mean.shape[0]):
    acc[i,0] = accuracy_self(out_seg_mean[i,:,:,1],ref[i,:,:,1])  # = RV
    acc[i,1] = accuracy_self(out_seg_mean[i,:,:,2],ref[i,:,:,2])  # = MYO
    acc[i,2] = accuracy_self(out_seg_mean[i,:,:,3],ref[i,:,:,3])  # = LV

mean_acc = np.mean(acc, axis=0)  
std_acc  = np.std(acc,  axis=0)
var_acc  = np.var(acc,  axis=0)

print('mean acc   = ',mean_acc)  
print('var acc    = ',  var_acc) 
print('std acc    = ',  std_acc) 

#%% IoU
jac = np.zeros((out_seg_mean.shape[0],3))

for i in range(0,out_seg_mean.shape[0]):
    jac[i,0] = jc(out_seg_mean[i,:,:,1],ref[i,:,:,1])  # = RV
    jac[i,1] = jc(out_seg_mean[i,:,:,2],ref[i,:,:,2])  # = MYO
    jac[i,2] = jc(out_seg_mean[i,:,:,3],ref[i,:,:,3])  # = LV

mean_jac = np.mean(jac, axis=0)  
std_jac  = np.std(jac,  axis=0)
var_jac  = np.var(jac,  axis=0)

print('mean jac   = ',  mean_jac)  
print('var jac    = ',  var_jac) 
print('std jac    = ',  std_jac) 


#%% MCC
mcc_cor = np.zeros((out_seg_mean.shape[0],3))

for i in range(0,out_seg_mean.shape[0]):
    mcc_cor[i,0] = mcc(out_seg_mean[i,:,:,1],ref[i,:,:,1])  # = RV
    mcc_cor[i,1] = mcc(out_seg_mean[i,:,:,2],ref[i,:,:,2])  # = MYO
    mcc_cor[i,2] = mcc(out_seg_mean[i,:,:,3],ref[i,:,:,3])  # = LV

mean_mcc = np.mean(mcc_cor, axis=0)  
std_mcc  = np.std(mcc_cor,  axis=0)
var_mcc  = np.var(mcc_cor,  axis=0)

print('mean mcc   = ',  mean_mcc)  
print('var mcc    = ',  var_mcc) 
print('std mcc    = ',  std_mcc) 

#%%




#%% Sensitivty/Recall
sen = np.zeros((out_seg_mean.shape[0],3))

for i in range(0,out_seg_mean.shape[0]):
    sen[i,0] = sensitivity(out_seg_mean[i,:,:,1],ref[i,:,:,1])  # = RV
    sen[i,1] = sensitivity(out_seg_mean[i,:,:,2],ref[i,:,:,2])  # = MYO
    sen[i,2] = sensitivity(out_seg_mean[i,:,:,3],ref[i,:,:,3])  # = LV

mean_sen = np.mean(sen, axis=0)  
std_sen  = np.std(sen,  axis=0)
var_sen  = np.var(sen,  axis=0)

print('mean sen   = ',  mean_sen)  
print('var sen    = ',  var_sen) 
print('std sen    = ',  std_sen) 


#%% Precision
prec = np.zeros((out_seg_mean.shape[0],3))

for i in range(0,out_seg_mean.shape[0]):
    prec[i,0] = precision(out_seg_mean[i,:,:,1],ref[i,:,:,1])  # = RV
    prec[i,1] = precision(out_seg_mean[i,:,:,2],ref[i,:,:,2])  # = MYO
    prec[i,2] = precision(out_seg_mean[i,:,:,3],ref[i,:,:,3])  # = LV

mean_prec = np.mean(prec, axis=0)  
std_prec  = np.std(prec,  axis=0)
var_prec  = np.var(prec,  axis=0)

print('mean prec   = ',  mean_prec)  
print('var prec    = ',  var_prec) 
print('std prec    = ',  std_prec) 

#%% Specificity
spec = np.zeros((out_seg_mean.shape[0],3))

for i in range(0,out_seg_mean.shape[0]):
    spec[i,0] = specificity(out_seg_mean[i,:,:,1],ref[i,:,:,1])  # = RV
    spec[i,1] = specificity(out_seg_mean[i,:,:,2],ref[i,:,:,2])  # = MYO
    spec[i,2] = specificity(out_seg_mean[i,:,:,3],ref[i,:,:,3])  # = LV

mean_spec = np.mean(spec, axis=0)  
std_spec  = np.std(spec,  axis=0)
var_spec  = np.var(spec,  axis=0)

print('mean spec   = ',  mean_spec)  
print('var spec    = ',  var_spec) 
print('std spec    = ',  std_spec) 


#%%
class_labels = ['RV', 'MYO', 'LV']
# Boxplot
plt.figure(figsize = (10,5), dpi=200)
plt.boxplot(dice, vert = False)
plt.yticks([1,2,3],['RV', 'MYO', 'LV'], fontsize = 14)
plt.xticks(fontsize = 14)
plt.title('Boxplot for Dice in Diastolic SD', fontsize = 20)

#%%%%%%%%%%%%%%%%%%%%%%% METRICS %%%%%%%%%%%%%%%%%%%%%

#%% Calculate volume for diastolic phase
#test_index = data_gt_ed[num_eval:num_test]

PATH_soft_dia_fold = path_out_soft# = '/Users/michalablicher/Desktop/Out_softmax_fold_avg_200dia_dice_10lclv.pt'

#PATH_soft_dia_fold = 'C:/Users/katrine/Desktop/Optuna/Out_softmax_fold_avg_200dia.pt'
#PATH_soft_dia_fold = 'C:/Users/katrine/Desktop/Optuna/Out_softmax_fold_avg_200dia.pt'
soft_dia_fold = torch.load(PATH_soft_dia_fold, map_location=torch.device(device))
#%%
soft_dia_mean = soft_dia_fold.mean(axis=0)
seg_dia_mean  = np.argmax(soft_dia_mean, axis=1)

seg_dia_oh    = torch.nn.functional.one_hot(torch.as_tensor(seg_dia_mean), num_classes=4).detach().cpu().numpy()
ref_dia_oh    = torch.nn.functional.one_hot(torch.as_tensor(Tensor(gt_test_ed_sub).to(torch.int64)), num_classes=4).detach().cpu().numpy()

test_index = len(p)

s = 0
target_vol_ed = np.zeros(test_index)
ref_vol_ed    = np.zeros(test_index)

for i in range(0,test_index):
    #print('patient nr.', i)
    for j in range(0, p[i]):
        #print('slice # ',j)
        target_vol_ed[i] += np.sum(seg_dia_oh[j+s,:,:,3])
        ref_vol_ed[i]    += np.sum(ref_dia_oh[j+s,:,:,3])
        #print('j+s = ',j+s)
        
    s += p[i] 
    #print('s= ',s)
   
#%% Calculate volume for systolic phase
PATH_soft_sys_fold = 'C:/Users/katrine/Desktop/Optuna/Out_softmax_fold_avg_200sys.pt'
soft_sys_fold = torch.load(PATH_soft_sys_fold, map_location=torch.device(device))

soft_sys_mean = soft_sys_fold.mean(axis=0)
seg_sys_mean  = np.argmax(soft_sys_mean, axis=1)

seg_sys_oh    = torch.nn.functional.one_hot(torch.as_tensor(seg_sys_mean), num_classes=4).detach().cpu().numpy()
ref_sys_oh    = torch.nn.functional.one_hot(torch.as_tensor(Tensor(gt_test_es_sub).to(torch.int64)), num_classes=4).detach().cpu().numpy()

test_index = len(p)

s = 0
target_vol_es = np.zeros(test_index)
ref_vol_es    = np.zeros(test_index)

for i in range(0,test_index):
    #print('patient nr.', i)
    for j in range(0, p[i]):
        #print('slice # ',j)
        target_vol_es[i] += np.sum(seg_sys_oh[j+s,:,:,3])
        ref_vol_es[i]    += np.sum(ref_sys_oh[j+s,:,:,3])
        #print('j+s = ',j+s)
        
    s += p[i] 
    #print('s= ',s)
     
#%% Calculate EF        
os.chdir('/Users/michalablicher/Documents/GitHub/Speciale2021')
#os.chdir("C:/Users/katrine/Documents/GitHub/Speciale2021")

from metrics import EF_calculation, dc, hd, jc, precision, mcc, recall, risk, sensitivity, specificity, true_negative_rate, true_positive_rate, positive_predictive_value, hd95, assd, asd, ravd, volume_correlation, volume_change_correlation, obj_assd, obj_asd, obj_fpr, obj_tpr

#%%
spacings = [1.4, 1.4, 8]

ef_ref    = EF_calculation(ref_vol_es, ref_vol_ed, spacings)
ef_target = EF_calculation(target_vol_es, target_vol_ed, spacings)


ef_m_ref = np.mean(ef_ref[0])
ef_m_tar = np.mean(ef_target[0])

print('EF ref  = ', ef_ref[0]) 
print('esv ref = ', ef_ref[1]) 
print('edv ref = ', ef_ref[2]) 

print('EF seg = ', ef_target[0]) 
print('esv seg = ', ef_target[1]) 
print('edv seg = ', ef_target[2]) 

#%%
slice = 169
c= 3

plt.imshow(seg_dia_oh[slice,:,:,c])
plt.imshow(seg_sys_oh[slice,:,:,c], alpha =0.5)

b = np.sum(ref_dia_oh[slice,:,:,c])


#%%
cor_edv = np.corrcoef(target_vol_ed,ref_vol_ed)

#%% E-map
emap = np.zeros((out_soft_mean.shape[0],out_soft_mean.shape[2],out_soft_mean.shape[3]))

test_slice = 28

# Slices 9, 28, 67, 84, 177, 186, 199, 248, 255, 265, 269, 312, 315


for i in range(0, emap.shape[0]):

    out_img  = out_soft_mean[i,:,:,:]
    #out_img = (out_soft[5,test_slice,:,:,:])
    entropy2 = scipy.stats.entropy(out_img)
    
    # Normalize 
    m_entropy   = np.max(entropy2)
    entropy     = entropy2/m_entropy
    emap[i,:,:] = entropy

emap = np.expand_dims(emap, axis=1)


#% Plot for visual inspection
# argmax seg + umap + GT


plt.figure(dpi=200, figsize=(20,10))
plt.subplot(3,1,1)
plt.imshow(out_seg_mean_am[test_slice,:,:])
plt.title('Seg. for slice: {}'.format(test_slice))

plt.subplot(3,1,2)
plt.imshow(emap[test_slice,0,:,:])
plt.title('Umap for slice: {}'.format(test_slice))

plt.subplot(3,1,3)
plt.imshow(gt_test_es_sub[test_slice,:,:])
plt.title('GT for slice: {}'.format(test_slice))

#%%
plt.figure(dpi=200)
plt.imshow(im_test_es_sub[test_slice,0,:,:])
plt.title('Img for slice: {}'.format(test_slice))


#%% Threshold prediction probabilities

seg_met = np.argmax(out_soft_mean, axis=1)

# Create Plot 
plt.figure(dpi=200)
plt.suptitle('Comparison of GT and predicted segmentation', fontsize=16 , y=0.8)

#n = 36 # anatomically incoherent 
n = 28 # anatomcally incoherent 
#n = 28 # totally incoherent 

plt.subplot(1, 3, 1)
plt.imshow(im_test_es_sub[test_slice,0,:,:])
plt.xticks(rotation=40, fontweight='light', fontsize=7,)
plt.yticks(horizontalalignment='right',fontweight='light', fontsize=7,)
plt.title('MRI', fontsize =10)

plt.subplot(1, 3, 2)
plt.imshow(gt_test_es_sub[n,:,:])
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
#os.chdir("C:/Users/katrine/Documents/GitHub/Speciale2021/")
#from metrics import dc, hd, risk, EF_calculation

dice = np.zeros(seg_met.shape[0])
haus = np.zeros(seg_met.shape[0])
fpos = np.zeros(seg_met.shape[0])
fneg = np.zeros(seg_met.shape[0])

for i in range(0,seg_met.shape[0]):
    #dice_m  = dc(seg_met[i,:,:],gt_test_es_sub[i,:,:])  
    #dice[i] = dice_m
    
    #haus_m  = hd(seg_met[i,:,:],gt_test_es_sub[i,:,:])  
    #haus[i] = haus_m
    
    fn_m, fp_m = risk(seg_met[i,:,:],gt_test_es_sub[i,:,:])  
    fneg[i] = fn_m
    fpos[i] = fp_m
        

#mean_dice = np.mean(dice)  
#mean_haus = np.mean(haus)
risk_measure = fpos+fneg


#print('mean overall dice = ',mean_dice)  
#print('mean overall haus = ',mean_haus)
print('mean overall risk = ',np.mean(risk_measure))

#print('Dice for test slice [n]      = ',dc(seg_met[n,:,:],gt_test_es_sub[n,:,:]) )
#print('Hausdorff for test slice [n] = ',hd(seg_met[n,:,:],gt_test_es_sub[n,:,:]) )
print('Risk measure test slice [n]  = ', risk_measure[n])

#%%

#out_seg_mean    = torch.nn.functional.one_hot(torch.as_tensor(out_seg_mean_am), num_classes=4).detach().cpu().numpy()

#ref = torch.nn.functional.one_hot(torch.as_tensor(Tensor(gt_test_es_sub).to(torch.int64)), num_classes=4).detach().cpu().numpy()


#%% Coverage threshold
os.chdir('/Users/michalablicher/Documents/GitHub/Speciale2021')
from uncertain_thres import get_seg_errors_mask, generate_thresholds
H = 128 
W = 128
pred   = Tensor(out_seg_mean).permute(0,3,1,2)
ref_e  = Tensor(ref).permute(0,3,1,2)
emap_e = np.squeeze(emap)

err_indices = get_seg_errors_mask(pred, gt_test_es_sub)
percentiles = generate_thresholds(pred, gt_test_es_sub, emap_e)

uncertain_voxels = np.zeros((len(percentiles),out_seg_mean.shape[0],H,W))
cov_slices       = np.zeros((out_seg_mean.shape[0],len(percentiles)))

for p, thres in enumerate(percentiles):
    for i in range(0,out_seg_mean.shape[0]):
        uncertain_voxels[p,i,:,:] = emap[i,:,:] >= thres
        cov_slices[i,p] = np.sum(uncertain_voxels[p,i,:,:])
    coverage = np.mean(cov_slices*1/(H*W)*100, axis=0)


#%% Risk measure

#thresholding the softmax for each of 4 channels by the percentile values
prob_thres = np.zeros((len(percentiles),337,4,H,W))
a  = np.zeros((337,4,H,W))
aa = np.zeros((len(percentiles),337,H,W))
risk_measure = np.zeros((len(percentiles),337))

#%%
for i in range(len(percentiles)):
    for k in range(0,10):#prob_thres.shape[1]):
        a[k,:,:,:]  = out_soft_mean[k,:,:,:] > percentiles[i]
        aa[i,:,:,:] = np.argmax(a,axis=1)
        
        fn_m, fp_m  = risk(aa[i,k,:,:],gt_test_es_sub[k,:,:])  
        
        risk_measure[i,k] = fp_m + fn_m
        
risk_m = np.mean(risk_measure, axis=1)


#%%
plt.figure(dpi=300)

thres_values = np.round([percentiles[50],percentiles[70], percentiles[80], percentiles[90]],8)
test_slice = 9

plt.suptitle('Comparison of uncertainty threholds', y=0.9)

plt.subplot(2,4,1)
plt.imshow(im_test_es_sub[test_slice,0,:,:])
plt.subplots_adjust(hspace = 0, wspace = 0.6)
plt.title('Original Im', fontsize=10)

plt.subplot(2,4,2)
plt.imshow(gt_test_es_sub[test_slice,:,:])
plt.title('GT seg.', fontsize=10)

plt.subplot(2,4,3)
plt.imshow(out_seg_mean_am[test_slice,:,:])
plt.title('Predicted seg.', fontsize=10)

plt.subplot(2,4,4)
plt.imshow(emap_e[test_slice,:,:])
plt.title('E-map', fontsize=10)
plt.colorbar(fraction=0.05)

plt.subplot(2,4,5)
plt.imshow(aa[50,test_slice,:,:])
plt.title('Thres: 8.27 e-12', fontsize=10)

plt.subplot(2,4,6)
plt.imshow(aa[70,test_slice,:,:])
plt.title('Thres: 7.40 e-07', fontsize=10)

plt.subplot(2,4,7)
plt.imshow(aa[90,test_slice,:,:])
plt.title('Thres: 1.90 e-04', fontsize=10)

plt.subplot(2,4,8)
plt.imshow(aa[99,test_slice,:,:])
plt.title('Thres: 3.55 e-02', fontsize=10)

#%%
# Risk-coverage curve
plt.figure(dpi=300)
plt.suptitle('Risk-Coverage curve', fontsize=16)
plt.plot(coverage[1:-1],risk_m[1:-1],'b.', label ='Unet-CE (e-map)')
plt.xlabel('Coverage [%]')
plt.ylabel('Risk (FP+FN)')



























