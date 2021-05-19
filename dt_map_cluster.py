#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 09:28:14 2021

@author: michalablicher
"""
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


#%% Specify directory
#os.chdir("C:/Users/katrine/Documents/GitHub/Speciale2021")
os.chdir('/Users/michalablicher/Documents/GitHub/Speciale2021')
#os.chdir('C:/Users/katrine/Documents/GitHub/Speciale2021')
from load_data_gt_im_sub import load_data_sub
user = 'M'
#user = 'K' 
data_im_es_DCM,  data_gt_es_DCM  = load_data_sub(user,'Systole','DCM')
data_im_es_HCM,  data_gt_es_HCM  = load_data_sub(user,'Systole','HCM')
data_im_es_MINF, data_gt_es_MINF = load_data_sub(user,'Systole','MINF')
data_im_es_NOR,  data_gt_es_NOR  = load_data_sub(user,'Systole','NOR')
data_im_es_RV,   data_gt_es_RV   = load_data_sub(user,'Systole','RV')

data_im_ed_DCM,  data_gt_ed_DCM  = load_data_sub(user,'Diastole','DCM')
data_im_ed_HCM,  data_gt_ed_HCM  = load_data_sub(user,'Diastole','HCM')
data_im_ed_MINF, data_gt_ed_MINF = load_data_sub(user,'Diastole','MINF')
data_im_ed_NOR,  data_gt_ed_NOR  = load_data_sub(user,'Diastole','NOR')
data_im_ed_RV,   data_gt_ed_RV   = load_data_sub(user,'Diastole','RV')

#%% BATCH GENERATOR
num_train_sub = 16 
num_eval_sub  = num_train_sub + 2
num_test_sub  = num_eval_sub + 2

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




#%% Load Model
#PATH_model = "C:/Users/katrine/Documents/Universitet/Speciale/Trained_Unet_CE_sys_nor20.pt"
#PATH_state = "C:/Users/katrine/Documents/Universitet/Speciale/Trained_Unet_CE_dia_nor_20e.p"
PATH_model_ed = '/Users/michalablicher/Desktop/Trained_Unet_CE_dia_sub_batch_100.pt'
PATH_model_es = '/Users/michalablicher/Desktop/Trained_Unet_CE_sys_sub_batch_100.pt'

# Load
unet_es = torch.load(PATH_model_es, map_location=torch.device('cpu'))
unet_ed = torch.load(PATH_model_ed, map_location=torch.device('cpu'))
#model.load_state_dict(torch.load(PATH_state))

#%% Running  models 
unet_es.eval()
out_trained_es = unet_es(Tensor(im_test_es_sub))
out_image_es    = out_trained_es["softmax"]
#%%
unet_ed.eval()
out_trained_ed = unet_ed(Tensor(im_test_ed_sub))
out_image_ed    = out_trained_ed["softmax"]

#%% One hot encoding data
seg_met_dia = np.argmax(out_image_ed.detach().numpy(), axis=1)

seg_dia = torch.nn.functional.one_hot(torch.as_tensor(seg_met_dia), num_classes=4).detach().numpy()
ref_dia = torch.nn.functional.one_hot(Tensor(gt_test_ed_sub).to(torch.int64), num_classes=4).detach().numpy()

seg_met_sys = np.argmax(out_image_es.detach().numpy(), axis=1)

seg_sys = torch.nn.functional.one_hot(torch.as_tensor(seg_met_sys), num_classes=4).detach().numpy()
ref_sys = torch.nn.functional.one_hot(Tensor(gt_test_es_sub).to(torch.int64), num_classes=4).detach().numpy()


#%% Cluster filter

from scipy.ndimage import label


seg_error_sys = abs(seg_sys - ref_sys)


cc_labels = np.zeros((seg_sys.shape))
n_cluster = np.zeros((seg_sys.shape[0]))

cluster_mask = np.zeros((seg_sys.shape))
cm_size      = np.zeros((seg_sys.shape))

min_size = 10
new_label_slice_sys = np.zeros_like(seg_sys)

for i in range(0, seg_sys.shape[0]):
    for j in range(0, seg_sys.shape[3]):
        cc_labels[i,:,:,j], n_cluster = label(seg_error_sys[i,:,:,j]) 
        #print(n_cluster) 
        for k in np.arange(1, n_cluster + 1):
            cluster_mask = cc_labels[i,:,:,j] == k
            
            cm_size = np.count_nonzero(cluster_mask)
            #print(cm_size)
            
            if cm_size >= min_size:
                new_label_slice_sys[i,cc_labels[i,:,:,j]== k ,j] = 1

#%% Show Results from clustering 
show_slice = 7
show_class = 1
plt.figure(dpi=2000)
plt.subplot(1,4,1)
plt.imshow(seg_sys[show_slice,:,:,show_class])
plt.title('Segmentation')
plt.subplot(1,4,2)
plt.imshow(seg_error_sys[show_slice,:,:,show_class])
plt.title('Error')
plt.subplot(1,4,3)
plt.imshow(new_label_slice_sys[show_slice,:,:,show_class])
plt.title('Cluster min 10')
plt.subplot(1,4,4)
plt.imshow(ref_sys[show_slice,:,:,show_class])
plt.title('Reference')


#%% Show Results from clustering 
show_slice = 11
show_class = 1
alpha = 0.5
for i in range(1,2):
    plt.figure(dpi=2000)
    plt.subplot(2,2,1)
    plt.subplots_adjust(hspace = 0.35)
    plt.imshow(seg_sys[show_slice,:,:,i])
    plt.imshow(im_test_es_sub[show_slice,0,:,:],alpha=alpha)
    plt.title('Segmentation', fontsize =10)
    plt.xticks(
    fontweight='light',
    fontsize=7)
    plt.yticks(
    fontweight='light',
    fontsize=7)
    plt.subplot(2,2,2)
    plt.subplots_adjust(hspace = 0.35, wspace = 0)
    plt.imshow(ref_sys[show_slice,:,:,i])
    plt.imshow(im_test_es_sub[show_slice,0,:,:],alpha=alpha)
    plt.title('Reference', fontsize =10)
    plt.xticks(
    fontweight='light',
    fontsize=7)
    plt.yticks(
    fontweight='light',
    fontsize=7)
    plt.subplot(2,2,3)
    plt.subplots_adjust(hspace = 0.35, wspace = 0)
    plt.imshow(seg_error_sys[show_slice,:,:,i])
    plt.imshow(im_test_es_sub[show_slice,0,:,:],alpha=alpha)
    plt.title('Difference', fontsize =10)
    plt.xticks(
    fontweight='light',
    fontsize=7)
    plt.yticks(
    fontweight='light',
    fontsize=7)
    plt.subplot(2,2,4)
    plt.subplots_adjust(hspace = 0.35, wspace = 0)
    plt.imshow(new_label_slice_sys[show_slice,:,:,i])
    plt.imshow(im_test_es_sub[show_slice,0,:,:],alpha=alpha)
    plt.title('Cluster min 10', fontsize =10)
    plt.xticks(
    fontweight='light',
    fontsize=7)
    plt.yticks(
    fontweight='light',
    fontsize=7)



#%% Cluster filter - Diastolic phase

from scipy.ndimage import label


seg_error_dia = abs(seg_dia - ref_dia)

cc_labels = np.zeros((seg_error_dia.shape))
n_cluster = np.zeros((seg_error_dia.shape[0]))

cluster_mask = np.zeros((seg_error_dia.shape))
cm_size      = np.zeros((seg_error_dia.shape))

min_size = 10
new_label_slice_dia = np.zeros_like(seg_error_dia)

n_cluster_1 = np.zeros((seg_error_dia.shape[0],seg_error_dia.shape[3]))
cm_size_1 = np.zeros((seg_error_dia.shape[0],seg_error_dia.shape[3]))

for i in range(0, seg_error_dia.shape[0]):
    for j in range(0, seg_error_dia.shape[3]):
        cc_labels[i,:,:,j], n_cluster = label(seg_error_dia[i,:,:,j]) 
        n_cluster_1[i,j] = n_cluster
        for k in np.arange(1, n_cluster + 1):
            cluster_mask = cc_labels[i,:,:,j] == k
            
            cm_size = np.count_nonzero(cluster_mask)
            cm_size_1[i,j] = cm_size
            #print(cm_size)
            
            if cm_size >= min_size:
                new_label_slice_dia[i,cc_labels[i,:,:,j]== k ,j] = 1
            #else: 
            #   new_label_slice_dia[cc_labels[i,:,:,j] == k] = 0

#%% Show Results from clustering 
show_slice = 10
show_class = 2
alpha = 0.5
for i in range(1,2):
    plt.figure(dpi=2000)
    plt.subplot(2,2,1)
    plt.subplots_adjust(hspace = 0.35)
    plt.imshow(seg_dia[show_slice,:,:,i])
    plt.imshow(im_test_ed_sub[show_slice,0,:,:],alpha=alpha)
    plt.title('Segmentation', fontsize =10)
    plt.xticks(
    fontweight='light',
    fontsize=7)
    plt.yticks(
    fontweight='light',
    fontsize=7)
    plt.subplot(2,2,2)
    plt.subplots_adjust(hspace = 0.35, wspace = 0)
    plt.imshow(ref_dia[show_slice,:,:,i])
    plt.imshow(im_test_ed_sub[show_slice,0,:,:],alpha=alpha)
    plt.title('Reference', fontsize =10)
    plt.xticks(
    fontweight='light',
    fontsize=7)
    plt.yticks(
    fontweight='light',
    fontsize=7)
    plt.subplot(2,2,3)
    plt.subplots_adjust(hspace = 0.35, wspace = 0)
    plt.imshow(seg_error_dia[show_slice,:,:,i])
    plt.imshow(im_test_ed_sub[show_slice,0,:,:],alpha=alpha)
    plt.title('Difference', fontsize =10)
    plt.xticks(
    fontweight='light',
    fontsize=7)
    plt.yticks(
    fontweight='light',
    fontsize=7)
    plt.subplot(2,2,4)
    plt.subplots_adjust(hspace = 0.35, wspace = 0)
    plt.imshow(new_label_slice_dia[show_slice,:,:,i])
    plt.imshow(im_test_ed_sub[show_slice,0,:,:],alpha=alpha)
    plt.title('Cluster min 10', fontsize =10)
    plt.xticks(
    fontweight='light',
    fontsize=7)
    plt.yticks(
    fontweight='light',
    fontsize=7)



#%% Function

def cluster_min(seg, ref, min_cluster_size):
     from scipy.ndimage import label
     seg_error = abs(seg - ref)
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
dia_new_label = cluster_min(seg_dia, ref_dia, 10)
sys_new_label = cluster_min(seg_sys, ref_sys, 10)

show_slice = 7
show_class = 2
plt.figure(dpi=2000)
plt.imshow(dia_new_label[show_slice,:,:,show_class])
plt.title('Diastolic: Cluster min 10')

plt.figure(dpi=2000)
plt.imshow(sys_new_label[show_slice,:,:,show_class])
plt.title('Systolic: Cluster min 10')





