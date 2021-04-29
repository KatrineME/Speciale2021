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
num_train = nor + 5#0
num_eval  = 3#0
num_test  = 10#0

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

#%%
roi_target_map = np.zeros((dt_es.shape))

for i in range(0, dt_es.shape[0]):
    for j in range(0, dt_es.shape[3]):
        roi_target_map[i,:,:,j] = np.logical_and(dt_es[i,:,:,j], sys_new_label[i,:,:,j])


#%%
test_slice = 14
class_title = ['Background','Right Ventricle','Myocardium','Left Ventricle']
plt.figure(dpi=200)

roi_mask = roi_target_map
roi_mask[roi_mask == 0]     = np.nan

for i in range (0,4):
    plt.subplot(1,4,i+1)
    plt.imshow(dt_es[test_slice,:,:,i])
    plt.imshow(sys_new_label[test_slice,:,:,i], alpha = 0.5)
    plt.imshow(roi_target_map[test_slice,:,:,i])
    plt.title(class_title[i], fontsize =10)
    plt.xticks([])
    plt.yticks([])




