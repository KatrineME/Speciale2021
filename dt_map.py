# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 11:49:19 2021

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

#!pip install torch-summary
#!pip install opencv-python

#%% Specify directory
os.chdir("C:/Users/katrine/Documents/GitHub/Speciale2021")
#os.chdir('/Users/michalablicher/Documents/GitHub/Speciale2021')

from load_data_gt_im import load_data

data_im_es, data_gt_es = load_data('K','Systole')
data_im_ed, data_gt_ed = load_data('K','Diastole')

#%% 
gt_es_flat = np.concatenate(data_gt_es).astype(None)
gt_ed_flat = np.concatenate(data_gt_ed).astype(None)

gt_es_oh = torch.nn.functional.one_hot(Tensor(gt_es_flat).to(torch.int64), num_classes=4).detach().numpy().astype(np.bool)
gt_ed_oh = torch.nn.functional.one_hot(Tensor(gt_ed_flat).to(torch.int64), num_classes=4).detach().numpy().astype(np.bool)

#%%

dt_es = np.zeros((gt_es_oh.shape))
dt_ed = np.zeros((gt_ed_oh.shape))

dt = np.zeros((gt_ed_oh.shape))

ref_border_es = np.zeros((gt_es_oh.shape))
ref_border_ed = np.zeros((gt_ed_oh.shape))

inside_obj_mask_es = np.zeros_like(gt_es_oh).astype(np.bool)
inside_obj_mask_ed = np.zeros_like(gt_ed_oh).astype(np.bool)

error_margin_inside  = 0 # VOXELS
error_margin_outside = 3 # VOXELS

dt_map = np.zeros((gt_es_oh.shape))

#for i in range(0, gt_es_oh.shape[0]):
for i in range(1, 4):
    for j in range(0, gt_es_oh.shape[3]):
        inside_voxels_indices   = binary_erosion(gt_es_oh[i,:,:,j], iterations=1)
        ref_border_es[i,:,:,j]  = np.logical_xor(gt_es_oh[i,:,:,j], inside_voxels_indices)
        ref_border_es           = ref_border_es.astype(bool)
        
        dt_es[i,:,:,j]          = distance_transform_edt(ref_border_es[i,:,:,j])
        dt[i,:,:,j]  = dt_es[i,:,:,j] 
        
        inside_obj_mask_es[i,inside_voxels_indices,j] = 1
        
        #inside_voxels_indices   = binary_erosion(gt_ed_oh[i,:,:,j], iterations=1)
        #ref_border_ed[i,:,:,j]  = np.logical_xor(gt_ed_oh[i,:,:,j], inside_voxels_indices)
        #dt_ed[i,:,:,j]          = distance_transform_edt(ref_border_ed)
        
        
        inside_obj_mask_ed[i,inside_voxels_indices,j] = 1
        
        
        
        #outside_obj_mask = np.logical_and(inside_obj_mask_es, ref_border_es)
        # surface border distance is always ZERO
        #dt_es[ref_border_es] = 0
        
        # inside structure: we subtract a fixed margin
        dt_es[i,inside_obj_mask_es[i,:,:,j],j] = dt_es[i,inside_obj_mask_es[i,:,:,j],j] - error_margin_inside
        
        # outside of target: structure we subtract a fixed margin.
        #dt_es[outside_obj_mask] = dt_es[outside_obj_mask] - error_margin_outside
        #dt_es[dt_es < 0] = 0
        
    print(i)
        #dt_map[i,:,:,j] = dt_es[i,:,:,j]

