# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 11:49:19 2021

@author: katrine
"""

import torch
import os
import numpy   as np
import matplotlib.pyplot as plt
from   scipy.ndimage.morphology import distance_transform_edt, binary_erosion
from   torch import Tensor

#%% Specify directory
os.chdir("C:/Users/katrine/Documents/GitHub/Speciale2021")
#os.chdir('/Users/michalablicher/Documents/GitHub/Speciale2021')

# Load data function
from load_data_gt_im import load_data

_ , data_gt_es = load_data('K','Systole')
_, data_gt_ed = load_data('K','Diastole')

#%% Load ground truth data 
gt_es_flat = np.concatenate(data_gt_es).astype(None)
gt_ed_flat = np.concatenate(data_gt_ed).astype(None)

# Onehot encode class channels
gt_es_oh = torch.nn.functional.one_hot(Tensor(gt_es_flat).to(torch.int64), num_classes=4).detach().numpy().astype(np.bool)
gt_ed_oh = torch.nn.functional.one_hot(Tensor(gt_ed_flat).to(torch.int64), num_classes=4).detach().numpy().astype(np.bool)

# Get rid of unecessary variables
del gt_ed_flat, gt_es_flat, data_gt_ed, data_gt_es

#%% Compute distance tarnsform maps

# Preallocate
dt_es = np.zeros((gt_es_oh.shape))
dt_ed = np.zeros((gt_ed_oh.shape))


ref_border_es = np.zeros((gt_es_oh.shape))
ref_border_ed = np.zeros((gt_ed_oh.shape))

inside_obj_mask_es = np.zeros_like(gt_es_oh).astype(np.bool)
inside_obj_mask_ed = np.zeros_like(gt_ed_oh).astype(np.bool)

outside_obj_mask_es = np.zeros_like(gt_es_oh).astype(np.bool)
outside_obj_mask_ed = np.zeros_like(gt_ed_oh).astype(np.bool)

# Specify tolerated distances
error_margin_inside  = 2 # VOXELS
error_margin_outside = 3 # VOXELS


for i in range(0, gt_es_oh.shape[0]):
    for j in range(0, gt_es_oh.shape[3]):
        
        # Find voxels inside object
        inside_voxels_indices_es = binary_erosion(gt_es_oh[i,:,:,j], iterations=1)
        inside_voxels_indices_ed = binary_erosion(gt_ed_oh[i,:,:,j], iterations=1)
        
        # Find border voxels
        ref_border_es[i,:,:,j]   = np.logical_xor(gt_es_oh[i,:,:,j], inside_voxels_indices_es)
        ref_border_ed[i,:,:,j]   = np.logical_xor(gt_ed_oh[i,:,:,j], inside_voxels_indices_ed)
        
        ref_border_es  = ref_border_es.astype(bool)
        ref_border_ed  = ref_border_ed.astype(bool)
        
        # Calculated euclidean distance to object for all voxels
        dt_es[i,:,:,j] = distance_transform_edt(~ref_border_es[i,:,:,j])
        dt_ed[i,:,:,j] = distance_transform_edt(~ref_border_ed[i,:,:,j])
        
        # save object masks
        inside_obj_mask_es[i,inside_voxels_indices_es,j] = 1
        inside_obj_mask_ed[i,inside_voxels_indices_ed,j] = 1        
        
        outside_obj_mask_es[i,:,:,j] = np.logical_and(~inside_obj_mask_es[i,:,:,j], ~ref_border_es[i,:,:,j])
        outside_obj_mask_ed[i,:,:,j] = np.logical_and(~inside_obj_mask_ed[i,:,:,j], ~ref_border_ed[i,:,:,j])
        
        # surface border distance is always ZERO
        dt_es[ref_border_es] = 0
        dt_ed[ref_border_ed] = 0
        
        # inside structure: we subtract a fixed margin
        dt_es[i,inside_obj_mask_es[i,:,:,j],j]  = dt_es[i,inside_obj_mask_es[i,:,:,j],j] - error_margin_inside
        dt_ed[i,inside_obj_mask_ed[i,:,:,j],j]  = dt_ed[i,inside_obj_mask_ed[i,:,:,j],j] - error_margin_inside
        
        # outside of target: structure we subtract a fixed margin.
        dt_es[i,outside_obj_mask_es[i,:,:,j],j] = dt_es[i, outside_obj_mask_es[i,:,:,j],j] - error_margin_outside
        dt_ed[i,outside_obj_mask_ed[i,:,:,j],j] = dt_ed[i, outside_obj_mask_ed[i,:,:,j],j] - error_margin_outside
        
        # Transform maps
        dt_es[dt_es < 0] = 0
        dt_ed[dt_ed < 0] = 0
    print(i)

"""
OBS: distance transform maps can't be computed for slices whitout GT annotation. This is the case for some apical and basal slices.
Segmentation errors for these slices should be included no matter of size and location.

"""

#%%
fig = plt.figure()

test_slice = 10

class_title = ['Background','Right Ventricle','Myocardium','Left Ventricle']
plt.figure(dpi=200, figsize=(15,15))
for i in range(0,4):
    plt.suptitle('Shown for slice %i' %test_slice, fontsize=20, y=0.8)
    plt.subplot(2, 4, i+1)
    plt.subplots_adjust(hspace = 0.0, wspace = 0.5)
    plt.imshow(gt_es_oh[test_slice,:,:,i])
    plt.title(class_title[i], fontsize =16)
    
    plt.subplot(2, 4, i+1+4)
    #plt.subplots_adjust(hspace = 0.0, wspace = 0.5)
    plt.imshow(dt_es[test_slice,:,:,i])
    plt.title(class_title[i], fontsize =16)
plt.show()   