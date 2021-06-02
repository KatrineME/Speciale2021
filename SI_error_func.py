# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 14:56:12 2021

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
#dt_es = dist_trans(gt_es_oh, 2,3)
#dt_ed = dist_trans(gt_ed_oh, 2,3)

#%% Filter size
def cluster_min(seg, ref, min_cluster_size):
     from scipy.ndimage import label
     seg_error = abs(seg - ref).astype('int32')
     cc_labels = np.zeros((seg_error.shape))
     n_cluster = np.zeros((seg_error.shape[0]))
    
     cluster_mask = np.zeros((seg_error.shape))
     cm_size      = np.zeros((seg_error.shape))
    
     min_size = 10
     new_label_slice = np.zeros_like(seg_error)
    
     n_cluster_1 = np.zeros((seg_error.shape[0],seg_error.shape[3]))
     cm_size_1 = np.zeros((seg_error.shape[0],seg_error.shape[3]))
     for i in range(0, seg_error.shape[0]):
        for j in range(0, seg_error.shape[3]):
            cc_labels[i,:,:,j], n_cluster = label(seg_error[i,:,:,j]) 
            n_cluster_1[i,j] = n_cluster
            for k in np.arange(1, n_cluster + 1):
                cluster_mask = cc_labels[i,:,:,j] == k
                
                cm_size = np.count_nonzero(cluster_mask)
                cm_size_1[i,j] = cm_size
                #print(cm_size)
                
                if cm_size >= min_size:
                    new_label_slice[i,cc_labels[i,:,:,j]== k ,j] = 1
                #else: 
                #   new_label_slice_dia[cc_labels[i,:,:,j] == k] = 0
     return new_label_slice
 
#%% Use function on dia or systolic
#dia_new_label = cluster_min(seg_dia, ref_dia, 10)
#sys_new_label = cluster_min(seg_sys, ref_sys, 10)