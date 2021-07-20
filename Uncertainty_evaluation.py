#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 11:50:59 2021

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


if torch.cuda.is_available():
    # Tensor = torch.cuda.FloatTensor
    device = 'cuda'
else:
    # Tensor = torch.FloatTensor
    device = 'cpu'
torch.cuda.manual_seed_all(808)


#%% Specify directory
user = 'M'

if user == 'M':
    os.chdir('/Users/michalablicher/Documents/GitHub/Speciale2021')
if user == 'K':
    os.chdir('C:/Users/katrine/Documents/GitHub/Speciale2021')
if user == 'GPU':
    os.chdir('/home/katrine/Speciale2021/Speciale2021')
    
    
from load_data_gt_im_sub_space import load_data_sub

data_im_ed_DCM,  data_gt_ed_DCM  = load_data_sub(user,'Diastole','DCM')
data_im_ed_HCM,  data_gt_ed_HCM  = load_data_sub(user,'Diastole','HCM')
data_im_ed_MINF, data_gt_ed_MINF = load_data_sub(user,'Diastole','MINF')
data_im_ed_NOR,  data_gt_ed_NOR  = load_data_sub(user,'Diastole','NOR')
data_im_ed_RV,   data_gt_ed_RV   = load_data_sub(user,'Diastole','RV')

#%% BATCH GENERATOR
num_train_sub = 12
num_eval_sub  = num_train_sub

num_train_res = num_eval_sub + 6
num_test_res  = num_train_res + 2

im_train_es_res = np.concatenate((np.concatenate(data_im_ed_DCM[num_eval_sub:num_train_res]).astype(None),
                                  np.concatenate(data_im_ed_HCM[num_eval_sub:num_train_res]).astype(None),
                                  np.concatenate(data_im_ed_MINF[num_eval_sub:num_train_res]).astype(None),
                                  np.concatenate(data_im_ed_NOR[num_eval_sub:num_train_res]).astype(None),
                                  np.concatenate(data_im_ed_RV[num_eval_sub:num_train_res]).astype(None)))

gt_train_es_res = np.concatenate((np.concatenate(data_gt_ed_DCM[num_eval_sub:num_train_res]).astype(None),
                                  np.concatenate(data_gt_ed_HCM[num_eval_sub:num_train_res]).astype(None),
                                  np.concatenate(data_gt_ed_MINF[num_eval_sub:num_train_res]).astype(None),
                                  np.concatenate(data_gt_ed_NOR[num_eval_sub:num_train_res]).astype(None),
                                  np.concatenate(data_gt_ed_RV[num_eval_sub:num_train_res]).astype(None)))


im_test_es_res = np.concatenate((np.concatenate(data_im_ed_DCM[num_train_res:num_test_res]).astype(None),
                                  np.concatenate(data_im_ed_HCM[num_train_res:num_test_res]).astype(None),
                                  np.concatenate(data_im_ed_MINF[num_train_res:num_test_res]).astype(None),
                                  np.concatenate(data_im_ed_NOR[num_train_res:num_test_res]).astype(None),
                                  np.concatenate(data_im_ed_RV[num_train_res:num_test_res]).astype(None)))

gt_test_es_res = np.concatenate((np.concatenate(data_gt_ed_DCM[num_train_res:num_test_res]).astype(None),
                                  np.concatenate(data_gt_ed_HCM[num_train_res:num_test_res]).astype(None),
                                  np.concatenate(data_gt_ed_MINF[num_train_res:num_test_res]).astype(None),
                                  np.concatenate(data_gt_ed_NOR[num_train_res:num_test_res]).astype(None),
                                  np.concatenate(data_gt_ed_RV[num_train_res:num_test_res]).astype(None)))
print('Data loaded+concat')

#%% U-Net
# LOAD THE SOFTMAX PROBABILITES OF THE 6 FOLD MODELS
#% Load softmax from ensemble models
#PATH_softmax_ensemble_unet = 'C:/Users/katrine/Desktop/Optuna/Out_softmax_fold_avg_test_ResNet.pt'
PATH_softmax_ensemble_unet = '/Users/michalablicher/Desktop/Out_softmax_fold_avg_dice_dia_150e_opt_test_ResNet.pt'

#PATH_softmax_ensemble_unet = '/home/katrine/Speciale2021/Speciale2021/Out_softmax_fold_avg.pt'
out_softmax_unet_fold = torch.load(PATH_softmax_ensemble_unet ,  map_location=torch.device(device))

# mean them over dim=0
#out_softmax_unet = out_softmax_unet_fold[:,252:,:,:,:].mean(axis=0)
out_softmax_unet = out_softmax_unet_fold.mean(axis=0)

#Argmax
seg_met = np.argmax(out_softmax_unet, axis=1)

# One hot encode
seg_oh = torch.nn.functional.one_hot(torch.as_tensor(seg_met), num_classes=4).detach().cpu().numpy()
ref_oh = torch.nn.functional.one_hot(Tensor(gt_test_es_res).to(torch.int64), num_classes=4).detach().cpu().numpy()

#%% E-map
import scipy.stats
emap = np.zeros((seg_oh.shape[0],seg_oh.shape[1],seg_oh.shape[2]))

for i in range(0, emap.shape[0]):

    out_img = out_softmax_unet[i,:,:,:]#.detach().cpu().numpy())
    entropy2 = scipy.stats.entropy(out_img)
    
    # Normalize 
    m_entropy   = np.max(entropy2)
    entropy     = entropy2/m_entropy
    emap[i,:,:] = entropy

emap = np.expand_dims(emap, axis=1)
#%%
#% Wrap all inputs together

im     = Tensor(im_test_es_res)
umap   = Tensor(emap)
seg    = Tensor(np.expand_dims(seg_met, axis = 1))
#%%
print('Sizes of concat: im, umap, seg',im.shape,umap.shape,seg.shape)

input_concat = torch.cat((im,umap,seg), dim=1)

image = 4*2

plt.figure(dpi=200, figsize = (15,15))

for i in range(0, 4):
    plt.subplot(4,4,1+i)
    #plt.subplots_adjust(wspace = 0.4)
    plt.imshow(im[i+image,0,:,:])
    plt.title('cMRI slice {}'.format(i+image)) 
    
    plt.subplot(4,4,5+i)
    plt.imshow(seg[i+image,0,:,:])
    plt.title('Segmentation') 
   
    plt.subplot(4,4,9+i)
    plt.imshow(umap[i+image,0,:,:])   
    plt.title('U-map') 
    
    plt.subplot(4,4,13+i)
    plt.imshow(gt_test_es_res[i+image,:,:])   
    plt.title('Reference') 

#%%
image = 78
difference = (gt_test_es_res[image,:,:]-seg[image,0,:,:].detach().numpy())
db = np.array(difference != 0)


plt.figure(dpi=200, figsize = (12,7))

plt.subplot(2,4,1)
#plt.subplots_adjust(wspace = 0.4)
plt.imshow(im[image,0,:,:])
plt.title('cMRI slice {}'.format(image))
plt.ylabel('Errors')

plt.subplot(2,4,2)
plt.imshow(seg[image,0,:,:])
plt.title('Segmentation') 
   
plt.subplot(2,4,3)
plt.imshow(gt_test_es_res[image,:,:])    
plt.title('Reference') 

plt.subplot(2,4,4)
plt.imshow(umap[image,0,:,:])   
plt.title('U-map') 


image = 79
#plt.figure(dpi=200, figsize = (12,7))

plt.subplot(2,4,5)
#plt.subplots_adjust(wspace = 0.4)
plt.imshow(im[image,0,:,:])
plt.title('cMRI slice {}'.format(image)) 
plt.ylabel('No Errors')

plt.subplot(2,4,6)
plt.imshow(seg[image,0,:,:])
plt.title('Segmentation') 
   
plt.subplot(2,4,7)
plt.imshow(gt_test_es_res[image,:,:])    
plt.title('Reference') 

plt.subplot(2,4,8)
plt.imshow(umap[image,0,:,:])   
plt.title('U-map') 
plt.colorbar(fraction=0.04)

#%%
plt.figure(dpi=200)
plt.imshow(umap_bin[image,0,:,:])   
plt.title('U-map') 
plt.colorbar()

#%%
os.chdir("/Users/michalablicher/Documents/GitHub/Speciale2021")
from metrics import accuracy_self, EF_calculation, dc, hd, jc, precision, mcc, recall, risk, sensitivity, specificity, true_negative_rate, true_positive_rate, positive_predictive_value, hd95, assd, asd, ravd, volume_correlation, volume_change_correlation, obj_assd, obj_asd, obj_fpr, obj_tpr

umap_bin = umap > 0.5
umap_bin_s = np.squeeze(umap_bin)

difference = (gt_test_es_res[:,:,:]-seg[:,0,:,:].detach().numpy())
db = (difference != 0)
db_b = (db.astype(np.float))

dice_umap = np.zeros(85)

for i in range(0,umap_bin.shape[0]):
    dice_umap[i] = dc(umap_bin_s[i,:,:].detach().numpy() ,db_b[i,:,:]) 

difference_u = (umap_bin_s[:,:,:].detach().numpy()-db_b[:,:,:])

print('min dice', np.min(dice_umap))
print('max dice', np.max(dice_umap))
print('mean dice', np.mean(dice_umap))
print('var dice', np.var(dice_umap))


#%%
image = 57

plt.figure(dpi=200, figsize=(7,2))
plt.subplot(1,3,1)
plt.imshow(umap[image,0,:,:])  
plt.title('U-map') 

plt.subplot(1,3,2)
plt.imshow(umap_bin_s[image,:,:])   
plt.title('Binarized U-map') 

plt.subplot(1,3,3)
plt.imshow(db[image,:,:])   
#plt.imshow(umap_bin_s[image,:,:],alpha=0.6)  
plt.title('Difference') 

#%%
plt.subplot(1,4,4)
plt.imshow(difference_u[image,:,:])     
plt.title('Difference') 

#%%
difference_u = (umap_bin_s[:,:,:].detach().numpy()-db_b[:,:,:])
 
plt.figure(dpi=200)
plt.imshow(db[image,:,:])   
plt.imshow(umap_bin_s[image,:,:],alpha=0.4)  
plt.title('Difference') 


#%%

dice_78 = dc(seg[78,:,:,:].detach().numpy() ,gt_test_es_res[78,:,:]) 

dice_79 = dc(seg[79,:,:,:].detach().numpy() ,gt_test_es_res[79,:,:]) 

print('dice_78', dice_78)
print('dice_79', dice_79)

#%%
"""
plt.figure(dpi=200, figsize = (15,15))
i = 0
image = 39
plt.subplot(4,1,1)
#plt.suptitle('Input for detection network', y=0.67, fontsize=25)
#plt.subplots_adjust(wspace = 0.4)
plt.imshow(im[i+image,0,:,:])
#plt.title('cMRI', fontsize=20) 

plt.subplot(4,1,2)
plt.imshow(umap[i+image,0,:,:])   
#plt.title('Segmentation', fontsize=20) 
   
plt.subplot(4,1,3)
plt.imshow(seg[i+image,0,:,:])
#plt.title('U-map', fontsize=20) 

plt.subplot(4,1,4)
plt.imshow(gt_test_es_res[i+image,:,:])   
#plt.title('U-map', fontsize=15) 
"""
#%%
plt.figure(dpi=200, figsize = (5,5))
plt.imshow(umap[i+image,0,:,:])   














