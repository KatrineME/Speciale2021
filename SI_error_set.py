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

# Load data function
from load_data_gt_im import load_data

data_im_es, data_gt_es = load_data('K','Systole')
data_im_ed, data_gt_ed = load_data('K','Diastole')

#%% Load test subjects
nor = 60
num_train = nor+5 #0
num_eval  = 3#0
num_test  =10#0

lim_eval  = num_train + num_eval
lim_test  = lim_eval + num_test

gt_es_flat = np.concatenate(data_gt_es[lim_eval:lim_test]).astype(None)
gt_ed_flat = np.concatenate(data_gt_ed[lim_eval:lim_test]).astype(None)

im_es_flat = np.concatenate(data_im_es[lim_eval:lim_test]).astype(None)
im_ed_flat= np.concatenate(data_im_ed[lim_eval:lim_test]).astype(None)

#%% Load model
PATH_model_es = "C:/Users/katrine/Documents/Universitet/Speciale/Trained_Unet_CE_sys_nor20.pt"
PATH_model_ed = "C:/Users/katrine/Documents/Universitet/Speciale/Trained_Unet_CE_dia_nor_20e.pt"

unet_es = torch.load(PATH_model_es, map_location=torch.device('cpu'))
unet_ed = torch.load(PATH_model_ed, map_location=torch.device('cpu'))

#%% Running  models 
unet_es.eval()
out_trained_es = unet_es(Tensor(im_es_flat))
out_image_es   = out_trained_es["softmax"]

unet_ed.eval()
out_trained_ed = unet_ed(Tensor(im_ed_flat))
out_image_ed   = out_trained_ed["softmax"]

#%% Onehot encode class channels
gt_es_oh = torch.nn.functional.one_hot(Tensor(gt_es_flat).to(torch.int64), num_classes=4).detach().numpy().astype(np.bool)
gt_ed_oh = torch.nn.functional.one_hot(Tensor(gt_ed_flat).to(torch.int64), num_classes=4).detach().numpy().astype(np.bool)

seg_met_sys = np.argmax(out_image_es.detach().numpy(), axis=1)
seg_met_dia = np.argmax(out_image_ed.detach().numpy(), axis=1)

seg_sys = torch.nn.functional.one_hot(torch.as_tensor(seg_met_sys), num_classes=4).detach().numpy()
seg_dia = torch.nn.functional.one_hot(torch.as_tensor(seg_met_dia), num_classes=4).detach().numpy()


#%% Distance transform maps
os.chdir("C:/Users/katrine/Documents/GitHub/Speciale2021")
from SI_error_func import dist_trans, cluster_min

error_margin_inside  = 2
error_margin_outside = 3

dt_es = dist_trans(gt_es_oh, error_margin_inside,error_margin_outside)
dt_ed = dist_trans(gt_ed_oh, error_margin_inside,error_margin_outside)


#%% filter cluster size
cluster_size = 10

dia_new_label = cluster_min(seg_dia, gt_ed_oh, cluster_size)
sys_new_label = cluster_min(seg_sys, gt_es_oh, cluster_size)

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
test_slice = 7
class_title = ['Background','Right Ventricle','Myocardium','Left Ventricle']
plt.figure(dpi=200)

for i in range (0,4):
    plt.subplot(3,4,i+1)
    plt.imshow(dt_ed[test_slice,:,:,i])
    cbar = plt.colorbar(fraction=0.06)
    cbar.ax.locator_params(nbins=5)
    cbar.ax.tick_params(labelsize=5)
    plt.imshow(im_ed_flat[test_slice,0,:,:], alpha =0.2)
    plt.title(class_title[i], fontsize =10)
    plt.subplots_adjust(hspace = 0.4, wspace = 0.5)
    plt.xticks([])
    plt.yticks([])
    
    if i == 0:
        plt.ylabel('Distance transform', fontsize=7)
    
    plt.subplot(3,4,i+1+4)
    plt.imshow(dia_new_label[test_slice,:,:,i])
    plt.imshow(im_ed_flat[test_slice,0,:,:], alpha =0.5)
    plt.xticks([])
    plt.yticks([])
    
    if i == 0:
        plt.ylabel('Filtered cluster', fontsize=7)
    
    plt.subplot(3,4,i+1+8)
    plt.imshow(roi_target_map_ed[test_slice,:,:,i])
    plt.imshow(im_ed_flat[test_slice,0,:,:], alpha =0.5)
    plt.xticks([])
    plt.yticks([])
    
    if i == 0:
        plt.ylabel('Resulting SI set', fontsize = 7)
        
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

plt.figure(dpi=2000)
plt.imshow(T_j[test_slice,:,:])
plt.title('Binary $t_j$ label', fontsize=14)
plt.xticks(np.arange(0,16, 2))

#%% Upsample
up      = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
Tj_temp = Tensor(np.expand_dims(T_j, axis=1))
up_im   = up(Tj_temp)

# Binarize
up_im[up_im >0] =1

#%%
# plot
plt.figure(dpi=200)
plt.imshow(up_im[test_slice,0,:,:])
plt.colorbar()
plt.imshow(im_ed_flat[test_slice,0,:,:], alpha= 0.6)
plt.imshow(seg_met_dia[test_slice,:,:], alpha= 0.6)
plt.title('Patches containing seg. failures', fontsize=14)



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

