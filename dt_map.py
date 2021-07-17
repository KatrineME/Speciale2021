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

user = 'K'
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

num_train_sub = 12
num_eval_sub = num_train_sub
num_test_sub = num_eval_sub + 8

"""
im_test_ed_sub = np.concatenate((np.concatenate(data_im_ed_DCM[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_im_ed_HCM[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_im_ed_MINF[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_im_ed_NOR[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_im_ed_RV[num_eval_sub:num_test_sub]).astype(None)))
"""
gt_test_ed_sub = np.concatenate((np.concatenate(data_gt_ed_DCM[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_gt_ed_HCM[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_gt_ed_MINF[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_gt_ed_NOR[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_gt_ed_RV[num_eval_sub:num_test_sub]).astype(None)))
"""
im_test_es_sub = np.concatenate((np.concatenate(data_im_es_DCM[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_im_es_HCM[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_im_es_MINF[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_im_es_NOR[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_im_es_RV[num_eval_sub:num_test_sub]).astype(None)))
"""
gt_test_es_sub = np.concatenate((np.concatenate(data_gt_es_DCM[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_gt_es_HCM[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_gt_es_MINF[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_gt_es_NOR[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_gt_es_RV[num_eval_sub:num_test_sub]).astype(None)))

#%% Load ground truth data 

# Onehot encode class channels
gt_es_oh = torch.nn.functional.one_hot(Tensor(gt_test_es_sub).to(torch.int64), num_classes=4).detach().numpy().astype(np.bool)
gt_ed_oh = torch.nn.functional.one_hot(Tensor(gt_test_ed_sub).to(torch.int64), num_classes=4).detach().numpy().astype(np.bool)

# Get rid of unecessary variables
#del gt_ed_flat, gt_es_flat, data_gt_ed, data_gt_es

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

#%% Define as function

def dist_trans(gt_oh, error_margin_inside, error_margin_outside):
    
    # Preallocate
    dt = np.zeros((gt_oh.shape))  
    ref_border       = np.zeros((gt_oh.shape))
    inside_obj_mask  = np.zeros_like(gt_oh).astype(np.bool)
    outside_obj_mask = np.zeros_like(gt_oh).astype(np.bool)

    for i in range(0, gt_oh.shape[0]):
        for j in range(0, gt_oh.shape[3]):
            
            # Find voxels inside object
            inside_voxels_indices = binary_erosion(gt_oh[i,:,:,j], iterations=1)

            # Find border voxels
            ref_border[i,:,:,j]   = np.logical_xor(gt_oh[i,:,:,j], inside_voxels_indices)
            
            ref_border = ref_border.astype(bool)
            
            # Calculated euclidean distance to object for all voxels
            dt[i,:,:,j] = distance_transform_edt(~ref_border[i,:,:,j])
            
            # save object masks
            inside_obj_mask[i, inside_voxels_indices, j] = 1               
            outside_obj_mask[i,:,:,j] = np.logical_and(~inside_obj_mask[i,:,:,j], ~ref_border[i,:,:,j])
            
            # surface border distance is always ZERO
            dt[ref_border] = 0
            
            # inside structure: we subtract a fixed margin
            dt[i,inside_obj_mask[i,:,:,j],j]  = dt[i, inside_obj_mask[i,:,:,j],j] - error_margin_inside
            
            # outside of target: structure we subtract a fixed margin.
            dt[i,outside_obj_mask[i,:,:,j],j] = dt[i, outside_obj_mask[i,:,:,j],j] - error_margin_outside
            
            # Transform maps
            dt[dt < 0] = 0
            
    return dt

#%% Test of function

dt_es = dist_trans(gt_es_oh, 2,3)
