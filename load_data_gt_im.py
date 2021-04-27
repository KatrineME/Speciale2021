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
    
    im = []
    gt = [] 
    centercrop = torchvision.transforms.CenterCrop((H,W))
    
    for i in range(0,num_patients):
        nimg = nib.load(frame_im[i])
        img  = nimg.get_fdata()

        n_gt = nib.load(frame_gt[i])
        anno = n_gt.get_fdata()
        
        im_slices     = img.shape[2]
        gt_slices     = anno.shape[2]
        
<<<<<<< HEAD
        n_gt = nib.load(frame_gt[i])
        anno = n_gt.get_fdata()
        
        im_slices      = img.shape[2]
=======
>>>>>>> 980cf3ce0d1a9a3badd99e1f76ab34e0c78dfb78
        centercrop_img = Tensor(np.zeros((H,W,im_slices)))
        centercrop_gt  = Tensor(np.zeros((H,W,gt_slices)))
        
        gt_slices     = anno.shape[2]
        centercrop_gt = Tensor(np.zeros((H,W,gt_slices)))
        
        for j in range(0,im_slices):
            centercrop_img[:,:,j] = centercrop(Tensor(img[:,:,j]))
            centercrop_gt[:,:,j] = centercrop(Tensor(anno[:,:,j]))
       
        in_image = np.expand_dims(centercrop_img,0)
        in_image = Tensor(in_image).permute(3,0,1,2).detach().numpy()
        im.append(in_image.astype(object))
<<<<<<< HEAD

=======
       
>>>>>>> 980cf3ce0d1a9a3badd99e1f76ab34e0c78dfb78
        in_gt = Tensor(centercrop_gt).permute(2,0,1).detach().numpy()
        gt.append(in_gt.astype(object))
        
    return im, gt
<<<<<<< HEAD
#%%


im_trial,gt_trial = load_data('M','Systole')


=======
#%% Example on how to call function:
    
# im_trial,gt_trial = load_data('K','Systole')
>>>>>>> 980cf3ce0d1a9a3badd99e1f76ab34e0c78dfb78

#%% How to concatenate lists of data:

#gt_sys = np.concatenate(gt_sys).astype(None)
#im_sys = np.concatenate(im_sys).astype(None)

#gt_dia = np.concatenate(gt_dia).astype(None)
#im_dia = np.concatenate(im_dia).astype(None)
