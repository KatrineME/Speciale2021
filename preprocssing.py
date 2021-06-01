# -*- coding: utf-8 -*-
"""
Created on Thu May 13 11:30:28 2021

@author: katrine
"""

import os
import cv2
import glob2
import torchvision
import scipy

from torch import Tensor
from PIL   import Image

import nibabel as nib
import numpy   as np
from torch import nn
import torch
import matplotlib.pyplot as plt

    
os.chdir("C:/Users/katrine/Documents/Universitet/Speciale/ACDC_training_data/training")

phase = 'Diastole'
frame_im = np.sort(glob2.glob('patient*/**/patient*_frame*[0-9].nii.gz'))
frame_gt = np.sort(glob2.glob('patient*/**/patient*_frame*[0-9]_gt.nii.gz'))


if phase == 'Diastole':
   phase = np.linspace(0,len(frame_im)-2,100).astype(int)
else:
   phase = np.linspace(1,len(frame_im)-1,100).astype(int)
        
# Divide frames
frame_im = frame_im[phase]
frame_gt = frame_gt[phase]

H = 128
W = H

num_slices = np.zeros(100)

center = np.zeros((100,2))
ori_resol = np.zeros((100,2))

gt_crop = [] #np.zeros((H,W,100))
im_crop = []

for i in range(0,100):
    #print('i =',i)
    nimg = nib.load(frame_im[i])
    img  = nimg.get_fdata()
    
    n_gt = nib.load(frame_gt[i])
    gt   = n_gt.get_fdata()
    
    gt_slices = gt.shape[2] - 1  # OBS: appical slices removed
        
    pad = 5  # padding added
    
    gt_p = np.zeros((gt.shape[0]+pad,gt.shape[1]+pad,gt_slices))
    img_p = np.zeros((img.shape[0]+pad,img.shape[1]+pad,gt_slices))
    
    for j in range(0,gt_slices):
        img_p[:,:,j] = np.pad(img[:,:,j],((pad,0),(pad,0)), 'constant', constant_values=0)
        gt_p[:,:,j]  = np.pad(gt[:,:,j],((pad,0),(pad,0)), 'constant', constant_values=0)
    
    c_slice   = int(np.floor(gt_slices/2))
    bin_gt    = np.zeros((gt_p.shape[0],gt_p.shape[1]))
    bin_gt[gt_p[:,:,c_slice] >= 1] = 1
    center[i,0],center[i,1]  = scipy.ndimage.center_of_mass(bin_gt)
    
    num_slices[i] = gt_slices
    
    cropped_gt = np.zeros((H,W,gt_slices))
    cropped_im = np.zeros((H,W,gt_slices))

    for j in range(0,gt_slices):
        top   = int(np.ceil(center[i,0] - (128/2)))
        bot   = int(np.ceil(center[i,0] + (128/2)))
        
        left  = int(np.ceil(center[i,1] - (128/2)))
        right = int(np.ceil(center[i,1] + (128/2)))
        
        cropped_gt[:,:,j] = gt_p[top:bot,left:right,j]
        cropped_im[:,:,j] = img_p[top:bot,left:right,j]
        
    in_image = np.expand_dims(cropped_im,0)
    in_image = Tensor(in_image).permute(3,0,1,2).detach().numpy()
    im_crop.append(in_image)
    
    in_gt = Tensor(cropped_gt).permute(2,0,1).detach().numpy()
    gt_crop.append(in_gt)
        
    