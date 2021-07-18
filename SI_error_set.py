# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 14:03:20 2021

@author: katrine
"""
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

from scipy.ndimage.morphology import distance_transform_edt, binary_erosion
from torch import nn
from torch import Tensor

#%% Specify directory
os.chdir("C:/Users/katrine/Documents/GitHub/Speciale2021")
#os.chdir('/Users/michalablicher/Documents/GitHub/Speciale2021')

user = 'K'

# Load data function
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
#%% Load model
PATH_model_es = "C:/Users/katrine/Desktop/Optuna/Final CV models/Out_softmax_fold_avg_150sys_dice_lclv.pt"
PATH_model_ed = "C:/Users/katrine/Desktop/Optuna/Final CV models/Out_softmax_fold_avg_150dia_dice_lclv.pt"

unet_es_soft = torch.load(PATH_model_es, map_location=torch.device('cpu'))
unet_ed_soft = torch.load(PATH_model_ed, map_location=torch.device('cpu'))

unet_es_mean    = unet_es_soft.mean(axis=0)
unet_ed_mean    = unet_ed_soft.mean(axis=0)

unet_es_mean_am = np.argmax(unet_es_mean, axis=1)
unet_ed_mean_am = np.argmax(unet_ed_mean, axis=1)

unet_es   = torch.nn.functional.one_hot(torch.as_tensor(unet_es_mean_am), num_classes=4).detach().cpu().numpy()
unet_ed   = torch.nn.functional.one_hot(torch.as_tensor(unet_ed_mean_am), num_classes=4).detach().cpu().numpy()


#%% Onehot encode class channels
gt_es_oh = torch.nn.functional.one_hot(Tensor(gt_test_es_sub).to(torch.int64), num_classes=4).detach().numpy().astype(np.bool)
gt_ed_oh = torch.nn.functional.one_hot(Tensor(gt_test_ed_sub).to(torch.int64), num_classes=4).detach().numpy().astype(np.bool)



#%% Distance transform maps
os.chdir("C:/Users/katrine/Documents/GitHub/Speciale2021")
from SI_error_func import dist_trans, cluster_min

error_margin_inside  = 2
error_margin_outside = 3

dt_es = dist_trans(gt_es_oh, error_margin_inside,error_margin_outside)
dt_ed = dist_trans(gt_ed_oh, error_margin_inside,error_margin_outside)


#%% filter cluster size
cluster_size = 10

dia_new_label = cluster_min(unet_ed, gt_ed_oh, cluster_size)
sys_new_label = cluster_min(unet_es, gt_es_oh, cluster_size)

#%% Apply both cluster size and dt map 
roi_target_map_es = np.zeros((dt_es.shape))
roi_target_map_ed = np.zeros((dt_ed.shape))

for i in range(0, dt_es.shape[0]):
    for j in range(0, dt_es.shape[3]):
        roi_target_map_es[i,:,:,j] = np.logical_and(dt_es[i,:,:,j], sys_new_label[i,:,:,j])

for i in range(0, dt_ed.shape[0]):
    for j in range(0, dt_es.shape[3]):
        roi_target_map_ed[i,:,:,j] = np.logical_and(dt_ed[i,:,:,j], dia_new_label[i,:,:,j])

#%% plot all results
test_slice = 29
class_title = ['Background','Right Ventricle','Myocardium','Left Ventricle']
plt.figure(dpi=200, figsize=(9,6))

alpha = 0.35
s = 10
for i in range (0,4):
    plt.subplot(3,4,i+1)
    plt.imshow(dt_ed[test_slice,:,:,i])
    cbar = plt.colorbar(fraction=0.06)
    cbar.ax.locator_params(nbins=5)
    cbar.ax.tick_params(labelsize=8)
    plt.imshow(im_test_ed_sub[test_slice,0,:,:], alpha =alpha*0.5)
    plt.title(class_title[i], fontsize =s)
    plt.subplots_adjust(hspace = 0.4, wspace = 0.5)
    #plt.xticks([])
    #plt.yticks([])
    
    if i == 0:
        plt.ylabel('Distance transform', fontsize=s)
    
    plt.subplot(3,4,i+1+4)
    plt.imshow(dia_new_label[test_slice,:,:,i])
    plt.imshow(im_test_ed_sub[test_slice,0,:,:], alpha =alpha)
    #plt.xticks([])
    #plt.yticks([])
    
    if i == 0:
        plt.ylabel('Filtered cluster', fontsize=s)
    
    plt.subplot(3,4,i+1+8)
    plt.imshow(roi_target_map_ed[test_slice,:,:,i])
    plt.imshow(im_test_ed_sub[test_slice,0,:,:], alpha =alpha)
    #plt.xticks([])
    #plt.yticks([])
    
    if i == 0:
        plt.ylabel('Resulting SI set', fontsize = s)
        
#%% plot all results
test_slice = 10
class_title = ['Background','Right Ventricle','Myocardium','Left Ventricle']
plt.figure(dpi=200, figsize=(11,8))

alpha = 0.35

s = 15

for i in range (0,4):
    plt.subplot(4,4,i*4+1)
    plt.imshow(dt_ed[test_slice,:,:,i])
    cbar = plt.colorbar(fraction=0.06)
    cbar.ax.locator_params(nbins=5)
    cbar.ax.tick_params(labelsize=s/2)
    plt.imshow(im_test_ed_sub[test_slice,0,:,:], alpha =alpha*0.5)
    
    if i == 0 or 4 or 8 or 12:
        plt.ylabel(class_title[i], fontsize =s)
    if i == 0:
        plt.title('Distance transform', fontsize=s)

    plt.subplot(4,4,i*4+2)
    plt.imshow(dia_new_label[test_slice,:,:,i])
    plt.imshow(im_test_ed_sub[test_slice,0,:,:], alpha =alpha)
    
    if i == 0:
        plt.title('Filtered clusters', fontsize=s)

    plt.subplot(4,4,i*4+3)
    plt.imshow(roi_target_map_ed[test_slice,:,:,i])
    plt.imshow(im_test_ed_sub[test_slice,0,:,:], alpha =alpha)

    if i == 0:
        plt.title('Resulting SI set', fontsize = s)

    plt.subplot(4,4,i*4+4)
    plt.imshow(im_test_ed_sub[test_slice,0,:,:])
    
    if i == 0:
        plt.title('Original cMRI', fontsize = s)


#%% Sample patches
patch_size = 8
patch_grid = int(roi_target_map_ed.shape[1]/patch_size)


# Preallocate
_temp  = np.zeros((patch_grid,patch_grid))
lin    = np.linspace(0,roi_target_map_ed.shape[1]-patch_size,patch_grid).astype(int)
_ctemp = np.zeros((patch_grid,patch_grid,roi_target_map_ed.shape[3]))
T_j    = np.zeros((roi_target_map_ed.shape[0],patch_grid,patch_grid,roi_target_map_ed.shape[3]))


for j in range (0,roi_target_map_ed.shape[0]):
    for c in range(0,4):
        for pp in range(0,16):
            for p, i in enumerate(lin):
                #_temp[pp,p] = np.count_nonzero(~np.isnan(roi_target_map_ed[j,lin[pp]:lin[pp]+8 , i:i+8, c]))
                _temp[pp,p] = np.count_nonzero(roi_target_map_ed[j,lin[pp]:lin[pp]+8 , i:i+8, c])
        _ctemp[:,:,c] = _temp
    T_j[j,:,:,:] = _ctemp


# BACKGROUND SEG FAILURES ARE REMOVED
T_j = T_j[:,:,:,1:] 

# Summing all tissue channels together
T_j = np.sum(T_j, axis = 3)

# Plot a final patch
# Binarize
T_j[T_j >= 1 ] = 1
#%%
plt.figure(dpi=200)
plt.imshow(T_j[test_slice,:,:])
plt.title('Binary $t_j$ label', fontsize=14)
plt.xticks(np.arange(0,16, 1))

#%% Upsample
up      = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
Tj_temp = Tensor(np.expand_dims(T_j, axis=1))
up_im   = up(Tj_temp)

# Binarize
up_im[up_im >0] =1

#%%
# plot
plt.figure(dpi=200)
plt.imshow(unet_ed_mean_am[test_slice,:,:])
plt.imshow(up_im[test_slice,0,:,:], alpha= 0.3)

#plt.colorbar()
#plt.imshow(im_test_ed_sub[test_slice,0,:,:], alpha= 0.6)

plt.title('Patches containing seg. errors', fontsize=14)



#%%
os.chdir("C:/Users/katrine/Documents/GitHub/Speciale2021")
from SI_error_set_func import SI_set

T_j_ny_dia = SI_set('K', 'dia', 0, 3)
T_j_ny_sys = SI_set('K', 'sys', 0, 3)


#%%
test_slice = 4

plt.subplot(1,2,1)
plt.imshow(T_j_ny_dia[test_slice,:,:])
plt.subplot(1,2,2)
plt.imshow(T_j_ny_sys[test_slice,:,:])

