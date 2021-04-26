# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 11:55:30 2021

@author: katrine
"""

def load_data(user,phase):
    #% Load packages
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
    import re
    
    #!pip install torch-summary
    #!pip install opencv-python
    #% Load paths
    
    if user == 'K':
        os.chdir("C:/Users/katrine/Documents/Universitet/Speciale/ACDC_training_data/training")
    else:
        os.chdir('/Users/michalablicher/Desktop/training')
    
    #frame_dia_im = np.sort(glob2.glob('patient*/**/patient*_frame01.nii.gz'))
    frame_im = np.sort(glob2.glob('patient*/**/patient*_frame*[0-9].nii.gz'))
    frame_gt = np.sort(glob2.glob('patient*/**/patient*_frame*[0-9]_gt.nii.gz'))
    
    if phase == 'Diastole':
        phase = np.linspace(0,len(frame_im)-2,100).astype(int)
    else:
        phase = np.linspace(1,len(frame_im)-1,100).astype(int)
    
    #% Divide frames
    frame_im = frame_im[phase]
    frame_gt = frame_gt[phase]
    
    #% Load images
    
    num_patients = len(frame_im)
    H = 128
    W = 128
    in_c = 1
    
    im     = []
    centercrop = torchvision.transforms.CenterCrop((H,W))
    
    for i in range(0,num_patients):
        nimg = nib.load(frame_im[i])
        img  = nimg.get_fdata()
        
        im_slices      = img.shape[2]
        centercrop_img = Tensor(np.zeros((H,W,im_slices)))
        
        for j in range(0,im_slices):
            centercrop_img[:,:,j] = centercrop(Tensor(img[:,:,j]))
       
        in_image = np.expand_dims(centercrop_img,0)
        in_image = Tensor(in_image).permute(3,0,1,2).detach().numpy()
        
        im.append(in_image.astype(object))
    
        
    #% Load gt
    gt = [] 
    for i in range(0,num_patients):
        n_gt = nib.load(frame_gt[i])
        anno  = n_gt.get_fdata()
        
        gt_slices     = anno.shape[2]
        centercrop_gt = Tensor(np.zeros((H,W,gt_slices)))
        
        for j in range(0,gt_slices):
            centercrop_gt[:,:,j] = centercrop(Tensor(anno[:,:,j]))
       
        in_gt = Tensor(centercrop_gt).permute(2,0,1).detach().numpy()
        gt.append(in_gt.astype(object))
        
    return im, gt
#%%


im_trial,gt_trial = load_data('M','Systole')



#%%
# OBS OBS OBS OBS OBS
# Images and gt are now lists and must be concatinated as np.array before plotting

gt_sys = np.concatenate(gt_sys).astype(None)
im_sys = np.concatenate(im_sys).astype(None)

gt_dia = np.concatenate(gt_dia).astype(None)
im_dia = np.concatenate(im_dia).astype(None)


#%%
plt.suptitle('Comparison of sys/dia data', y=1)

slice = 

plt.subplot(2,2,1)
plt.imshow(im_dia[slice,0,:,:])
plt.title('Im: Diastole', fontsize=10)
plt.subplots_adjust(hspace = 0.40, wspace = 0)
plt.subplot(2,2,2)
plt.imshow(gt_dia[slice,:,:])
plt.title('GT: Diastole', fontsize=10)
plt.subplot(2,2,3)
plt.imshow(im_sys[slice,0,:,:])
plt.title('Im: Systole', fontsize=10)
plt.subplot(2,2,4)
plt.imshow(gt_sys[slice,:,:])
plt.title('GT: Systole', fontsize=10)
