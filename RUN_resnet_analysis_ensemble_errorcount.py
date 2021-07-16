#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 10:27:26 2021

@author: michalablicher
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 11:41:06 2021

@author: katrine
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

#path_out_soft = '/Users/michalablicher/Desktop/Out_softmax_fold_avg_100dia_dice_lclv.pt'
path_out_soft = 'C:/Users/katrine/Desktop/Optuna/Final CV models/Out_softmax_fold_avg_150sys_dice_lclv.pt'

out_soft = torch.load(path_out_soft ,  map_location=torch.device(device))

#%%
#Plot softmax probabilities for a single slice
test_slice = 31
alpha = 0.4

fig = plt.figure()

class_title = ['Background','Right Ventricle','Myocardium','Left Ventricle']
ref_dia = torch.nn.functional.one_hot(Tensor(gt_test_ed_sub).to(torch.int64), num_classes=4)
plt.figure(dpi=200, figsize=(18,32))

w = 0.1

for fold_model in range (0,6):
    out_img_ed = np.squeeze(out_soft[fold_model,test_slice,:,:,:])
    seg_met_dia = np.argmax(out_soft[fold_model,:,:,:], axis=1)
    seg_dia = torch.nn.functional.one_hot(torch.as_tensor(seg_met_dia), num_classes=4)
    
    #Reference annotation
    plt.suptitle('Systolic phase: test image at slice %i for CV folds' %test_slice, fontsize=30, y=0.9)
    plt.subplot(7, 4, 1)
    plt.subplots_adjust(hspace = 0.05, wspace = w)
    plt.imshow(ref_dia[test_slice,:,:,0])
    plt.imshow(im_test_ed_sub[test_slice,0,:,:],alpha=alpha)
    plt.ylabel('Reference', fontsize=16)
    plt.title('Background', fontsize=16)
    
    plt.subplot(7, 4, 2)
    plt.subplots_adjust(hspace = 0.05, wspace = w)
    plt.imshow(ref_dia[test_slice,:,:,1])
    plt.imshow(im_test_ed_sub[test_slice,0,:,:],alpha=alpha)
    plt.title('Right ventricle', fontsize=16)
    
    plt.subplot(7, 4, 3)
    plt.subplots_adjust(hspace = 0.05, wspace = w)
    plt.imshow(ref_dia[test_slice,:,:,2])
    plt.imshow(im_test_ed_sub[test_slice,0,:,:],alpha=alpha)
    plt.title('Myocardium', fontsize=16)
    
    plt.subplot(7, 4, 4)
    plt.subplots_adjust(hspace = 0.05, wspace = w)
    plt.imshow(ref_dia[test_slice,:,:,3])
    plt.imshow(im_test_ed_sub[test_slice,0,:,:],alpha=alpha)
    plt.title('Left ventricle', fontsize=16)
    
    
    #CV model segmentations
    plt.subplot(7, 4, 1+4*(fold_model+1))
    plt.subplots_adjust(hspace = 0.05, wspace = w)
    plt.imshow(seg_dia[test_slice,:,:,0])
    plt.imshow(im_test_ed_sub[test_slice,0,:,:],alpha=alpha)
    plt.ylabel('CV fold {}'.format(fold_model), fontsize=16)
    
    plt.subplot(7, 4, 2+4*(fold_model+1))
    plt.subplots_adjust(hspace = 0.05, wspace = w)
    plt.imshow(seg_dia[test_slice,:,:,1])
    plt.imshow(im_test_ed_sub[test_slice,0,:,:],alpha=alpha)
    
    plt.subplot(7, 4, 3+4*(fold_model+1))
    plt.subplots_adjust(hspace = 0.05, wspace = w)
    plt.imshow(seg_dia[test_slice,:,:,2])
    plt.imshow(im_test_ed_sub[test_slice,0,:,:],alpha=alpha)
    
    plt.subplot(7, 4, 4+4*(fold_model+1))
    plt.subplots_adjust(hspace = 0.05, wspace = w)
    plt.imshow(seg_dia[test_slice,:,:,3])
    plt.imshow(im_test_ed_sub[test_slice,0,:,:],alpha=alpha)

plt.show()  

#%% Mean + argmax + one hot

out_soft_mean   = out_soft.mean(axis=0)
out_seg_mean_am = np.argmax(out_soft_mean, axis=1)
out_seg_mean    = torch.nn.functional.one_hot(torch.as_tensor(out_seg_mean_am), num_classes=4).detach().cpu().numpy()

ref = torch.nn.functional.one_hot(torch.as_tensor(Tensor(gt_test_es_sub).to(torch.int64)), num_classes=4).detach().cpu().numpy()
#%% Plot of input data 
test_slice = 125

plt.figure(dpi=200)
plt.suptitle('Input data for systole + diastole')
plt.subplot(2,2,1)
plt.imshow(im_test_es_sub[test_slice,0,:,:])

plt.subplot(2,2,2)
plt.imshow(im_test_ed_sub[test_slice,0,:,:])

plt.subplot(2,2,3)
plt.imshow(gt_test_es_sub[test_slice,:,:])

plt.subplot(2,2,4)
plt.imshow(gt_test_ed_sub[test_slice,:,:])


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

    print('inside', inside.shape)    
    return inside #torch.sum(Tensor(inside))/(128*128*32)#.cuda()

out_seg_per = Tensor(out_seg_mean).permute(0,3,1,2)   # dim: [B,C,H,W]
lv_neigh = lv_loss(out_seg_per)

c_non = np.count_nonzero(lv_neigh, axis = (1,2)) # number of error pixels in each slice

cnon_slice = np.count_nonzero(c_non) # number of slices with erros 
print('Number of slices with errors:', cnon_slice)
print('Percentage of slices with errors:', (cnon_slice/len(c_non))*100,'%')
print('Number of errornous neighbour pixels:', c_non.sum())

#%%
def soft_dice_loss(y_true, y_pred):
     """ Calculate soft dice loss for each class
        y_pred = bs x c x h x w
        y_true = bs x c x h x w (one hot)
     """
     eps = 1e-6
     w = 1/8
     
     numerator   = 2. * torch.sum(y_pred * y_true, (2,3)) 
     denominator = torch.sum((torch.square(y_pred) + torch.square(y_true)), (2,3))
     h =  1 - ((numerator + eps) / (denominator + eps)) 
     c = Tensor(np.expand_dims(np.array([1*w,2*w,4*w,1*w]), axis=0))
     return torch.sum(c*h), c*h, h

d, ch,h = soft_dice_loss(out_seg_per, Tensor(ref).permute(0,3,1,2))
print(d/337)
print(d.shape)

#%%
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
w = 0.1
h = 0.3
test_slice = 31
plt.figure(dpi=200)
plt.suptitle('Systolic - Averaged model for test image at slice: {}'.format(test_slice))

plt.subplot(2,2,1)
plt.subplots_adjust(hspace = h, wspace = w)
plt.imshow(out_seg_mean[test_slice,:,:,0])
plt.imshow(im_test_es_sub[test_slice,0,:,:],alpha=alpha)
plt.title('Background', fontsize=10)

plt.subplot(2,2,2)
plt.subplots_adjust(hspace = h, wspace = w)
plt.imshow(out_seg_mean[test_slice,:,:,1])
plt.imshow(im_test_es_sub[test_slice,0,:,:],alpha=alpha)
plt.title('Right ventricle', fontsize=10)

plt.subplot(2,2,3)
plt.subplots_adjust(hspace = h, wspace = w)
plt.imshow(out_seg_mean[test_slice,:,:,2])
plt.imshow(im_test_es_sub[test_slice,0,:,:],alpha=alpha)
plt.title('Myocardium', fontsize=10)

plt.subplot(2,2,4)
plt.subplots_adjust(hspace = h, wspace = w)
plt.imshow(out_seg_mean[test_slice,:,:,3])
plt.imshow(im_test_es_sub[test_slice,0,:,:],alpha=alpha)
plt.title('Left ventricle', fontsize=10)

#%% Metrics
os.chdir("C:/Users/katrine/Documents/GitHub/Speciale2021")
#os.chdir("/Users/michalablicher/Documents/GitHub/Speciale2021")
from metrics import accuracy_self, EF_calculation, dc, hd, jc, precision, mcc, recall, risk, sensitivity, specificity, true_negative_rate, true_positive_rate, positive_predictive_value, hd95, assd, asd, ravd, volume_correlation, volume_change_correlation, obj_assd, obj_asd, obj_fpr, obj_tpr

dice = np.zeros((out_seg_mean.shape[0],3))
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
print('var dice    = ',  var_dice) 
print('std dice    = ',  std_dice) 


print('mean haus   = ',mean_haus)
print('var haus    = ', var_haus) 

print('std haus    = ',  std_haus)
print('std haus95  = ',  std_haus95)

print('mean haus95 = ',mean_haus95)
print('var haus95  = ', var_haus95)

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

#%% MCC
mcc_cor = np.zeros((out_seg_mean.shape[0],3))

for i in range(0,out_seg_mean.shape[0]):
    mcc_cor[i,0] = mcc(out_seg_mean[i,:,:,1],ref[i,:,:,1])  # = RV
    mcc_cor[i,1] = mcc(out_seg_mean[i,:,:,2],ref[i,:,:,2])  # = MYO
    mcc_cor[i,2] = mcc(out_seg_mean[i,:,:,3],ref[i,:,:,3])  # = LV

mean_mcc = np.mean(mcc_cor, axis=0)  
std_mcc  = np.std(mcc_cor,  axis=0)
var_mcc  = np.var(mcc_cor,  axis=0)

print('mean mcc   = ',mean_mcc)  
print('var mcc    = ',  var_mcc) 
print('std mcc    = ',  std_mcc) 


#%% Sensitivty/Recall
sen = np.zeros((out_seg_mean.shape[0],3))

for i in range(0,out_seg_mean.shape[0]):
    sen[i,0] = sensitivity(out_seg_mean[i,:,:,1],ref[i,:,:,1])  # = RV
    sen[i,1] = sensitivity(out_seg_mean[i,:,:,2],ref[i,:,:,2])  # = MYO
    sen[i,2] = sensitivity(out_seg_mean[i,:,:,3],ref[i,:,:,3])  # = LV

mean_sen = np.mean(sen, axis=0)  
std_sen  = np.std(sen,  axis=0)
var_sen  = np.var(sen,  axis=0)

print('mean sen   = ',mean_sen)  
print('var sen    = ',  var_sen) 
#print('std sen    = ',  std_sen) 
#%% Specificity
spec = np.zeros((out_seg_mean.shape[0],3))

for i in range(0,out_seg_mean.shape[0]):
    spec[i,0] = specificity(out_seg_mean[i,:,:,1],ref[i,:,:,1])  # = RV
    spec[i,1] = specificity(out_seg_mean[i,:,:,2],ref[i,:,:,2])  # = MYO
    spec[i,2] = specificity(out_seg_mean[i,:,:,3],ref[i,:,:,3])  # = LV

mean_spec = np.mean(spec, axis=0)  
std_spec  = np.std(spec,  axis=0)
var_spec  = np.var(spec,  axis=0)

print('mean spec   = ',mean_spec)  
print('var spec    = ',  var_spec) 
print('std spec    = ',  std_spec) 
#%% Precision
prec = np.zeros((out_seg_mean.shape[0],3))

for i in range(0,out_seg_mean.shape[0]):
    prec[i,0] = precision(out_seg_mean[i,:,:,1],ref[i,:,:,1])  # = RV
    prec[i,1] = precision(out_seg_mean[i,:,:,2],ref[i,:,:,2])  # = MYO
    prec[i,2] = precision(out_seg_mean[i,:,:,3],ref[i,:,:,3])  # = LV

mean_prec = np.mean(prec, axis=0)  
std_prec  = np.std(prec,  axis=0)
var_prec  = np.var(prec,  axis=0)

print('mean prec   = ',mean_prec)  
print('var prec    = ',  var_prec) 
#print('std prec    = ',  std_prec)
#%% Boxplots
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

#PATH_soft_dia_fold = path_out_soft# = '/Users/michalablicher/Desktop/Out_softmax_fold_avg_200dia_dice_10lclv.pt'

PATH_soft_dia_fold = 'C:/Users/katrine/Desktop/Optuna/Final CV models/Out_softmax_fold_avg_150dia_dice.pt'
#PATH_soft_dia_fold = 'C:/Users/katrine/Desktop/Optuna/Out_softmax_fold_avg_200dia.pt'
soft_dia_fold = torch.load(PATH_soft_dia_fold, map_location=torch.device(device))

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
    
print('Target vol dia: ',target_vol_ed)
print('Reference vol dia: ',ref_vol_ed)
   
#%% Calculate volume for systolic phase
PATH_soft_sys_fold = 'C:/Users/katrine/Desktop/Optuna/Final CV models/Out_softmax_fold_avg_150sys_dice.pt'
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
     
print('Target vol sys: ',target_vol_es)
print('Reference vol sys: ',ref_vol_es)
#%% Calculate EF        
#os.chdir('/Users/michalablicher/Documents/GitHub/Speciale2021')
os.chdir("C:/Users/katrine/Documents/GitHub/Speciale2021")

from metrics import EF_calculation, dc, hd, jc, precision, mcc, recall, risk, sensitivity, specificity, true_negative_rate, true_positive_rate, positive_predictive_value, hd95, assd, asd, ravd, volume_correlation, volume_change_correlation, obj_assd, obj_asd, obj_fpr, obj_tpr

#%% Ejection fraction
spacings = [1.4, 1.4, 8]

EF_ref    = EF_calculation(ref_vol_es, ref_vol_ed, spacings)
EF_target = EF_calculation(target_vol_es, target_vol_ed, spacings)


EF_m_ref = np.mean(EF_ref[0])
EF_m_tar = np.mean(EF_target[0])

print('EF mean target: ', EF_m_tar)
print('EF mean reference: ', EF_m_ref)


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

for i in range(0, emap.shape[0]):

    out_img  = out_soft_mean[i,:,:,:]
    entropy2 = scipy.stats.entropy(out_img)
    
    # Normalize 
    m_entropy   = np.max(entropy2)
    entropy     = entropy2/m_entropy
    emap[i,:,:] = entropy

emap = np.expand_dims(emap, axis=1)


#% Plot for visual inspection
# argmax seg + umap + GT
test_slice = 329

plt.figure(dpi=200, figsize=(20,10))
plt.subplot(3,1,1)
plt.imshow(out_seg_mean_am[test_slice,:,:])
plt.title('Seg. for slice: {}'.format(test_slice))

plt.subplot(3,1,2)
plt.imshow(emap[test_slice,0,:,:])
plt.title('Umap for slice: {}'.format(test_slice))

plt.subplot(3,1,3)
plt.imshow(gt_test_ed_sub[test_slice,:,:])
plt.title('GT for slice: {}'.format(test_slice))


