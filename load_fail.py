# -*- coding: utf-8 -*-
"""
Created on Mon May 24 13:10:39 2021

@author: katrine
"""
import torch
import os

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

#%%
os.chdir("C:/Users/katrine/Documents/Universitet/Speciale/ACDC_training_data/training")   # Local directory katrine
#os.chdir('/Users/michalablicher/Desktop/training')     # Local directory michala
#os.chdir("/home/michala/training")                      # Server directory michala


    #frame_dia_im = np.sort(glob2.glob('patient*/**/patient*_frame01.nii.gz'))
frame_im = np.sort(glob2.glob('patient*/**/patient*_frame*[0-9].nii.gz'))
frame_gt = np.sort(glob2.glob('patient*/**/patient*_frame*[0-9]_gt.nii.gz'))
        
phase = 'Diastole'
diagnose = 'DCM'
    
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
                
    im_slices  = img.shape[2]
    gt_slices  = anno.shape[2]
        
    centercrop_img = Tensor(np.zeros((H,W,im_slices)))
    centercrop_gt  = Tensor(np.zeros((H,W,gt_slices)))

            
    for j in range(0,im_slices):
        print(img.shape)
        center_img = centercrop(Tensor(img[:,:,j]))
        print(center_img.shape)
        centercrop_img[:,:,j] = (center_img-torch.mean(center_img)) / torch.std(center_img)
        #centercrop_img[:,:,j]  = Tensor(cv2.normalize(center_img.detach().numpy(), None, 255, 0, cv2.NORM_MINMAX))
        
        """ NORM_MINMAX: The minimum pixel value will be mapped to the minimum output value(alpha), 
            the maximum pixel value will be mapped to the maximum output value(beta). 
            With linear scaling for everything in between """
            
        centercrop_gt[:,:,j]  = centercrop(Tensor(anno[:,:,j]))
  
    in_image = np.expand_dims(centercrop_img,0)
    in_image = Tensor(in_image).permute(3,0,1,2).detach().numpy()
    im.append(in_image.astype(object))
    
    in_gt = Tensor(centercrop_gt).permute(2,0,1).detach().numpy()
    gt.append(in_gt.astype(object))
    
#%%
import torch
import os
import cv2
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


os.chdir('C:/Users/katrine/Documents/GitHub/Speciale2021')
from load_data_gt_im_sub import load_data_sub

data_im_es_DCM,  data_gt_es_DCM  = load_data_sub('K','Systole','DCM')


#%%
data_im_ed_DCM,  data_gt_ed_DCM  = load_data_sub('K','Diastole','DCM')
data_im_ed_HCM,  data_gt_ed_HCM  = load_data_sub('K','Diastole','HCM')
data_im_ed_MINF, data_gt_ed_MINF = load_data_sub('K','Diastole','MINF')
data_im_ed_NOR,  data_gt_ed_NOR  = load_data_sub('K','Diastole','NOR')
data_im_ed_RV,   data_gt_ed_RV   = load_data_sub('K','Diastole','RV')

num_train_sub = 16 
num_eval_sub = num_train_sub + 2
num_test_sub = num_eval_sub + 2

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

ref_dia = torch.nn.functional.one_hot(Tensor(gt_test_ed_sub).to(torch.int64), num_classes=4).detach().numpy()
#%%
ref = Tensor(ref_dia).permute(0,3,1,2)

y_ref = torch.sum(ref, (2,3))
y_pred = Tensor(im_test_ed_sub) - 4
epsilon = 0.000001

a = y_ref

if not a.detach().numpy().all():
    #loss_c = -1* torch.sum( torch.log(1-y_pred + epsilon),(2,3))
    print(a)
    print('Something is zero')
else:
    print(a)
    print('Nothing is zero')
    
    
    
    
    
    
    
    
    
    
    
    
    
    




