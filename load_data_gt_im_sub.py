# -*- coding: utf-8 -*-
"""
Created on Tue May 11 13:04:26 2021

@author: katrine
"""
def load_data_sub(user, phase, diagnose):

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
    
    
    # Load paths
        
    if user == 'K':
        os.chdir("C:/Users/katrine/Documents/Universitet/Speciale/ACDC_training_data/training")
    elif user == 'GPU':
        os.chdir("/home/michala/training")                      # Server directory michala
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
    H  = 128
    W  = 128
     
    im = []
    gt = [] 
    centercrop = torchvision.transforms.CenterCrop((H,W))
    
    num_case = 20  # Number of patients within each subgroup
    
    DCM  = np.linspace(0, num_case-1, num_case).astype(int)
    HCM  = np.linspace(DCM[-1]+1,  DCM[-1]+num_case,  num_case).astype(int)
    MINF = np.linspace(HCM[-1]+1,  HCM[-1]+num_case,  num_case).astype(int)
    NOR  = np.linspace(MINF[-1]+1, MINF[-1]+num_case, num_case).astype(int)
    RV   = np.linspace(NOR[-1]+1,  NOR[-1]+num_case,  num_case).astype(int)
    
    if diagnose == 'DCM':
        sub = DCM
    elif diagnose == 'HCM':
        sub = HCM
    elif diagnose == 'MINF':
        sub = MINF
    elif diagnose == 'NOR':
        sub = NOR
    else:
        sub = RV
    
    
    for i in sub:
        nimg = nib.load(frame_im[i])
        img  = nimg.get_fdata()
    
        n_gt = nib.load(frame_gt[i])
        anno = n_gt.get_fdata()
                
        im_slices  = img.shape[2]-1
        gt_slices  = anno.shape[2]-1
        
        centercrop_img = Tensor(np.zeros((H,W,im_slices)))
        centercrop_gt  = Tensor(np.zeros((H,W,gt_slices)))
                
        #gt_slices     = anno.shape[2]-1
        #centercrop_gt = Tensor(np.zeros((H,W,gt_slices)))
            
        for j in range(0,im_slices):
            center_img = centercrop(Tensor(img[:,:,j]))
            #centercrop_img[:,:,j] = (center_img-torch.mean(center_img)) / torch.std(center_img)
            #centercrop_img[:,:,j] = Tensor(cv2.normalize(center_img.detach().numpy(), None, 255, 0, cv2.NORM_MINMAX))   
            
            centercrop_gt[:,:,j]  = centercrop(Tensor(anno[:,:,j]))
        
        in_image = np.expand_dims(centercrop_img,0)
        in_image = Tensor(in_image).permute(3,0,1,2).detach().numpy()
        im.append(in_image.astype(object))
    
        in_gt = Tensor(centercrop_gt).permute(2,0,1).detach().numpy()
        gt.append(in_gt.astype(object))
        
    return im, gt




















