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

#!pip install torch-summary
#!pip install opencv-python

#%% Specify directory
#os.chdir("C:/Users/katrine/Documents/GitHub/Speciale2021")
#os.chdir('/Users/michalablicher/Documents/GitHub/Speciale2021')

from load_data_gt_im import load_data

data_im_es, data_gt_es = load_data('M','Systole')
data_im_ed, data_gt_ed = load_data('M','Diastole')


#%% Cluster analysis - Test Data
nor = 60
num_train = nor + 5#0
num_eval  = 3#0
num_test  = 10#0

lim_eval  = num_train + num_eval
lim_test  = lim_eval + num_test

im_flat_test_es = np.concatenate(data_im_es[lim_eval:lim_test]).astype(None)
gt_flat_test_es = np.concatenate(data_gt_es[lim_eval:lim_test]).astype(None)

im_flat_test_ed = np.concatenate(data_im_ed[lim_eval:lim_test]).astype(None)
gt_flat_test_ed = np.concatenate(data_gt_ed[lim_eval:lim_test]).astype(None)


#%% Load Model
#PATH_model = "C:/Users/katrine/Documents/GitHub/Speciale2021/trained_Unet_testtest.pt"
#PATH_state = "C:/Users/katrine/Documents/GitHub/Speciale2021/trained_Unet_testtestate.pt"

PATH_model_es = '/Users/michalablicher/Desktop/Trained_Unet_CE_sys_nor20.pt'
PATH_model_ed = '/Users/michalablicher/Desktop/Trained_Unet_CE_dia_nor.pt'

# Load
unet_es = torch.load(PATH_model_es, map_location=torch.device('cpu'))
unet_ed = torch.load(PATH_model_ed, map_location=torch.device('cpu'))
#model.load_state_dict(torch.load(PATH_state))

#%% Running  models 
unet_es.eval()
out_trained_es = unet_es(Tensor(im_flat_test_es))
out_image_es    = out_trained_es["softmax"]

unet_ed.eval()
out_trained_ed = unet_ed(Tensor(im_flat_test_ed))
out_image_ed    = out_trained_ed["softmax"]

#%% One hot encoding data
seg_met_dia = np.argmax(out_image_ed.detach().numpy(), axis=1)

seg_dia = torch.nn.functional.one_hot(torch.as_tensor(seg_met_dia), num_classes=4).detach().numpy()
ref_dia = torch.nn.functional.one_hot(Tensor(gt_flat_test_ed).to(torch.int64), num_classes=4).detach().numpy()

seg_met_sys = np.argmax(out_image_es.detach().numpy(), axis=1)

seg_sys = torch.nn.functional.one_hot(torch.as_tensor(seg_met_sys), num_classes=4).detach().numpy()
ref_sys = torch.nn.functional.one_hot(Tensor(gt_flat_test_es).to(torch.int64), num_classes=4).detach().numpy()


#%% Cluster filter

from scipy.ndimage import label


seg_error = seg_sys - ref_sys


cc_labels = np.zeros((seg_sys.shape))
n_cluster = np.zeros((seg_sys.shape[0]))

cluster_mask = np.zeros((seg_sys.shape))
cm_size      = np.zeros((seg_sys.shape))

min_size = 10
new_label_slice = np.zeros_like(seg_sys)

for i in range(0, seg_sys.shape[0]):
    for j in range(0, seg_sys.shape[3]):
        cc_labels[i,:,:,j], n_cluster = label(seg_error[i,:,:,j]) 
        #print(n_cluster) 
        for k in np.arange(1, n_cluster + 1):
            cluster_mask = cc_labels[i,:,:,j] == k
            
            cm_size = np.count_nonzero(cluster_mask)
            #print(cm_size)
            
            if cm_size >= min_size:
                new_label_slice[cc_labels == k] = 1
            else: 
                new_label_slice[cc_labels == k] = 0

#%% Show Results from clustering 
show_slice = 65
show_class = 2
plt.figure(dpi=2000)
plt.subplot(1,4,1)
plt.imshow(seg_sys[show_slice,:,:,show_class])
plt.title('Segmentation')
plt.subplot(1,4,2)
plt.imshow(seg_error[show_slice,:,:,show_class])
plt.title('Error')
plt.subplot(1,4,3)
plt.imshow(new_label_slice[show_slice,:,:,show_class])
plt.title('Cluster min 10')
plt.subplot(1,4,4)
plt.imshow(ref_sys[show_slice,:,:,show_class])
plt.title('Reference')























