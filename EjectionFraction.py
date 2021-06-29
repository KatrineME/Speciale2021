# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 09:03:18 2021

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
import torchsummary

#from skimage.transform import resize
from torch import nn
from torch import Tensor
import scipy.ndimage


if torch.cuda.is_available():
    # Tensor = torch.cuda.FloatTensor
    device = 'cuda'
else:
    # Tensor = torch.FloatTensor
    device = 'cpu'
torch.cuda.manual_seed_all(808)

#%% Specify directory
if device == 'cuda':
    user = 'GPU'
else:
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

#%% BATCH GENERATOR
num_train_sub = 12
num_eval_sub = num_train_sub
num_test_sub = num_eval_sub + 8

im_test_dia_sub = np.concatenate((np.concatenate(data_im_ed_DCM[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_im_ed_HCM[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_im_ed_MINF[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_im_ed_NOR[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_im_ed_RV[num_eval_sub:num_test_sub]).astype(None)))

gt_test_dia_sub = np.concatenate((np.concatenate(data_gt_ed_DCM[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_gt_ed_HCM[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_gt_ed_MINF[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_gt_ed_NOR[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_gt_ed_RV[num_eval_sub:num_test_sub]).astype(None)))

im_test_sys_sub = np.concatenate((np.concatenate(data_im_es_DCM[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_im_es_HCM[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_im_es_MINF[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_im_es_NOR[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_im_es_RV[num_eval_sub:num_test_sub]).astype(None)))

gt_test_sys_sub = np.concatenate((np.concatenate(data_gt_es_DCM[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_gt_es_HCM[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_gt_es_MINF[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_gt_es_NOR[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_gt_es_RV[num_eval_sub:num_test_sub]).astype(None)))
#%% Load 
path_soft_dia = 'C:/Users/katrine/Desktop/Optuna/Final CV models/Out_softmax_fold_avg_100dia_dice_lv.pt'
path_soft_sys = 'C:/Users/katrine/Desktop/Optuna/Final CV models/Out_softmax_fold_avg_100sys_dice_lv.pt'

soft_dia = torch.load(path_soft_dia ,  map_location=torch.device(device))
soft_sys = torch.load(path_soft_sys ,  map_location=torch.device(device))

#%% Mean + argmax + one hot

soft_dia_mean    = soft_dia.mean(axis=0)
soft_dia_mean_am = np.argmax(soft_dia_mean, axis=1)
out_seg_dia_mean = torch.nn.functional.one_hot(torch.as_tensor(soft_dia_mean_am), num_classes=4).detach().cpu().numpy()

ref_dia = torch.nn.functional.one_hot(torch.as_tensor(Tensor(gt_test_dia_sub).to(torch.int64)), num_classes=4).detach().cpu().numpy()

soft_sys_mean    = soft_sys.mean(axis=0)
soft_sys_mean_am = np.argmax(soft_sys_mean, axis=1)
out_seg_sys_mean = torch.nn.functional.one_hot(torch.as_tensor(soft_sys_mean_am), num_classes=4).detach().cpu().numpy()

ref_sys = torch.nn.functional.one_hot(torch.as_tensor(Tensor(gt_test_sys_sub).to(torch.int64)), num_classes=4).detach().cpu().numpy()

#%%%%%%%%%%%%%%%%%%%%%%% METRICS %%%%%%%%%%%%%%%%%%%%%
# Slices per patient
p = []    # Slices per patient

for i in range(0,8):
    p.append(data_gt_ed_DCM[num_eval_sub:num_test_sub][i].shape[0])
for i in range(0,8):
    p.append(data_gt_ed_HCM[num_eval_sub:num_test_sub][i].shape[0])
for i in range(0,8):
    p.append(data_gt_ed_MINF[num_eval_sub:num_test_sub][i].shape[0])
for i in range(0,8):
    p.append(data_gt_ed_NOR[num_eval_sub:num_test_sub][i].shape[0])
for i in range(0,8):
    p.append(data_gt_ed_RV[num_eval_sub:num_test_sub][i].shape[0])
    
#%% Volume DIASTOLIC
test_index = len(p)

s = 0
target_vol_dia = np.zeros(test_index)
ref_vol_dia    = np.zeros(test_index)

target_vol_dia_RV = np.zeros(test_index)
ref_vol_dia_RV    = np.zeros(test_index)

for i in range(0,test_index):
    #print('patient nr.', i)
    for j in range(0, p[i]):
        #print('slice # ',j)
        target_vol_dia[i] += np.sum(out_seg_dia_mean[j+s,:,:,3])
        ref_vol_dia[i]    += np.sum(ref_dia[j+s,:,:,3])
        
        target_vol_dia_RV[i] += np.sum(out_seg_dia_mean[j+s,:,:,1])
        ref_vol_dia_RV[i]    += np.sum(ref_dia[j+s,:,:,1])
        #print('j+s = ',j+s)
    s += p[i] 
#%% Volume SYSTOLIC
test_index = len(p)

s = 0
target_vol_sys = np.zeros(test_index)
ref_vol_sys    = np.zeros(test_index)

target_vol_sys_RV = np.zeros(test_index)
ref_vol_sys_RV    = np.zeros(test_index)

for i in range(0,test_index):
    #print('patient nr.', i)
    for j in range(0, p[i]):
        #print('slice # ',j)
        target_vol_sys[i] += np.sum(out_seg_sys_mean[j+s,:,:,3])
        ref_vol_sys[i]    += np.sum(ref_sys[j+s,:,:,3])

        target_vol_sys_RV[i] += np.sum(out_seg_sys_mean[j+s,:,:,1])
        ref_vol_sys_RV[i]    += np.sum(ref_sys[j+s,:,:,1])
        #print('j+s = ',j+s)
    s += p[i] 

#%% EJECTION FREACTION
os.chdir("C:/Users/katrine/Documents/GitHub/Speciale2021")

from metrics import EF_calculation, dc, hd, jc, precision, mcc, recall, risk, sensitivity, specificity, true_negative_rate, true_positive_rate, positive_predictive_value, hd95, assd, asd, ravd, volume_correlation, volume_change_correlation, obj_assd, obj_asd, obj_fpr, obj_tpr

#%%
spacings = [1.4, 1.4, 8] #mm/voxel

EF_ref    = EF_calculation(ref_vol_sys, ref_vol_dia, spacings)
EF_target = EF_calculation(target_vol_sys, target_vol_dia, spacings)

EF_ref_RV    = EF_calculation(ref_vol_sys, ref_vol_dia, spacings)
EF_target_RV = EF_calculation(target_vol_sys, target_vol_dia, spacings)


ef_m_ref = np.mean(EF_ref[0])
ef_m_tar = np.mean(EF_target[0])

print('EF ref = \n',EF_ref[0]) 
print('EF seg = \n',EF_target[0]) 

print('End-sys volume ref = \n', EF_ref[1]) 
print('End-dia volume ref = \n', EF_ref[2]) 

print('End-sys volume seg = \n', EF_target[1]) 
print('End-dia volume seg = \n', EF_target[2]) 

#%% Correlation
cor_dia = np.corrcoef(target_vol_dia, ref_vol_dia)
cor_sys = np.corrcoef(target_vol_sys, ref_vol_sys)

cor_dia_RV = np.corrcoef(target_vol_dia_RV, ref_vol_dia_RV)
cor_sys_RV = np.corrcoef(target_vol_sys_RV, ref_vol_sys_RV)
#%% LV
print('Correlation diastole =', cor_dia[1,0]) 
print('Correlation systole  =', cor_sys[1,0]) 
#%%
print('Correlation diastole =', cor_dia_RV) 
print('Correlation systole  =', cor_sys_RV) 