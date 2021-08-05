# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 14:40:38 2021

@author: katrine
"""
#%% Load packages
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
import skimage
from skimage import measure

#from skimage.transform import resize
from torch import nn
from torch import Tensor
import scipy.ndimage
import scipy.stats
import torchsummary



user = 'K'
if user == 'M':
    os.chdir('/Users/michalablicher/Documents/GitHub/Speciale2021')
if user == 'K':
    os.chdir('C:/Users/katrine/Documents/GitHub/Speciale2021')
if user == 'GPU':
    os.chdir('/home/michala/Speciale2021/Speciale2021')

 
from load_data_gt_im_sub_space import load_data_sub

phase = 'Systole'

data_im_es_DCM,  data_gt_es_DCM  = load_data_sub(user,phase,'DCM')
data_im_es_HCM,  data_gt_es_HCM  = load_data_sub(user,phase,'HCM')
data_im_es_MINF, data_gt_es_MINF = load_data_sub(user,phase,'MINF')
data_im_es_NOR,  data_gt_es_NOR  = load_data_sub(user,phase,'NOR')
data_im_es_RV,   data_gt_es_RV   = load_data_sub(user,phase,'RV')

phase = 'Diastole'

data_im_ed_DCM,  data_gt_ed_DCM  = load_data_sub(user,phase,'DCM')
data_im_ed_HCM,  data_gt_ed_HCM  = load_data_sub(user,phase,'HCM')
data_im_ed_MINF, data_gt_ed_MINF = load_data_sub(user,phase,'MINF')
data_im_ed_NOR,  data_gt_ed_NOR  = load_data_sub(user,phase,'NOR')
data_im_ed_RV,   data_gt_ed_RV   = load_data_sub(user,phase,'RV')

num_train_sub = 12
num_eval_sub = num_train_sub
num_test_sub = num_eval_sub + 8
#%% BATCH GENERATOR   TEST   TEST   TEST   TEST   TEST   TEST   TEST   TEST   TEST   TEST   TEST   TEST   TEST   TEST   TEST   TEST   TEST   TEST   TEST   TEST   TEST

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

#%% BATCH GENERATOR   TRAINING    TRAINING   TRAINING   TRAINING   TRAINING   TRAINING   TRAINING   TRAINING   TRAINING   TRAINING   TRAINING   TRAINING   TRAINING   TRAINING
im_train_ed_sub = np.concatenate((np.concatenate(data_im_ed_DCM[0:num_train_sub]).astype(None),
                                  np.concatenate(data_im_ed_HCM[0:num_train_sub]).astype(None),
                                  np.concatenate(data_im_ed_MINF[0:num_train_sub]).astype(None),
                                  np.concatenate(data_im_ed_NOR[0:num_train_sub]).astype(None),
                                  np.concatenate(data_im_ed_RV[0:num_train_sub]).astype(None)))

gt_train_ed_sub = np.concatenate((np.concatenate(data_gt_ed_DCM[0:num_train_sub]).astype(None),
                                  np.concatenate(data_gt_ed_HCM[0:num_train_sub]).astype(None),
                                  np.concatenate(data_gt_ed_MINF[0:num_train_sub]).astype(None),
                                  np.concatenate(data_gt_ed_NOR[0:num_train_sub]).astype(None),
                                  np.concatenate(data_gt_ed_RV[0:num_train_sub]).astype(None)))

im_train_es_sub = np.concatenate((np.concatenate(data_im_es_DCM[0:num_train_sub]).astype(None),
                                  np.concatenate(data_im_es_HCM[0:num_train_sub]).astype(None),
                                  np.concatenate(data_im_es_MINF[0:num_train_sub]).astype(None),
                                  np.concatenate(data_im_es_NOR[0:num_train_sub]).astype(None),
                                  np.concatenate(data_im_es_RV[0:num_train_sub]).astype(None)))

gt_train_es_sub = np.concatenate((np.concatenate(data_gt_es_DCM[0:num_train_sub]).astype(None),
                                  np.concatenate(data_gt_es_HCM[0:num_train_sub]).astype(None),
                                  np.concatenate(data_gt_es_MINF[0:num_train_sub]).astype(None),
                                  np.concatenate(data_gt_es_NOR[0:num_train_sub]).astype(None),
                                  np.concatenate(data_gt_es_RV[0:num_train_sub]).astype(None)))

#%% DIASTOLE
ref_dia = torch.nn.functional.one_hot(Tensor(gt_test_ed_sub).to(torch.int64), num_classes=4)

s = torch.sum(ref_dia,axis=(1,2))
slices = s.detach().numpy()

d = ((s/(128*128))*100).detach().numpy()

bgr = np.mean(d[:,0])
RV  = np.mean(d[:,1])
MYO = np.mean(d[:,2])
LV  = np.mean(d[:,3])

print('Number of empty RV in diatole (test):', ((s.shape[0]-np.count_nonzero(s[:,1]))/s.shape[0])*100)

ref_dia = torch.nn.functional.one_hot(Tensor(gt_train_ed_sub).to(torch.int64), num_classes=4)

s = torch.sum(ref_dia,axis=(1,2))
slices = s.detach().numpy()

d = ((s/(128*128))*100).detach().numpy()

bgr = np.mean(d[:,0])
RV  = np.mean(d[:,1])
MYO = np.mean(d[:,2])
LV  = np.mean(d[:,3])

print('Number of empty RV in diatole (train):', ((s.shape[0]-np.count_nonzero(s[:,1]))/s.shape[0])*100)
#%% SYSTOLE
ref_sys = torch.nn.functional.one_hot(Tensor(gt_test_es_sub).to(torch.int64), num_classes=4)

s = torch.sum(ref_sys,axis=(1,2))
slices = s.detach().numpy()

d = ((s/(128*128))*100).detach().numpy()

bgr = np.mean(d[:,0])
RV  = np.mean(d[:,1])
MYO = np.mean(d[:,2])
LV  = np.mean(d[:,3])

print('Number of empty RV in systole (test):', ((s.shape[0]-np.count_nonzero(s[:,1]))/s.shape[0])*100)

ref_sys = torch.nn.functional.one_hot(Tensor(gt_train_es_sub).to(torch.int64), num_classes=4)

s = torch.sum(ref_sys,axis=(1,2))
slices = s.detach().numpy()

d = ((s/(128*128))*100).detach().numpy()

bgr = np.mean(d[:,0])
RV  = np.mean(d[:,1])
MYO = np.mean(d[:,2])
LV  = np.mean(d[:,3])

print('Number of empty RV in systole (train):', ((s.shape[0]-np.count_nonzero(s[:,1]))/s.shape[0])*100)