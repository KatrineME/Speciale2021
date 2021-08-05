# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 12:42:51 2021

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

#from skimage.transform import resize
from torch import nn
from torch import Tensor
import scipy.ndimage
import scipy.stats
import torchsummary


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

print('Data loaded+concat')

#%%
path_out_1 = 'C:/Users/katrine/Desktop/Optuna/Final CV models/Out_softmax_fold_avg_150sys_dice_opt.pt'
path_out_2 = 'C:/Users/katrine/Desktop/Optuna/Final CV models/Out_softmax_fold_avg_150sys_dice_lclv_opt.pt'

out_soft1 = torch.load(path_out_1 ,  map_location=torch.device(device))
out_soft2 = torch.load(path_out_2 ,  map_location=torch.device(device))


#% Mean + argmax + one hot

out_soft_mean1   = out_soft1.mean(axis=0)
out_seg_mean_am1 = np.argmax(out_soft_mean1, axis=1)
out_seg_mean1    = torch.nn.functional.one_hot(torch.as_tensor(out_seg_mean_am1), num_classes=4).detach().cpu().numpy()

out_soft_mean2   = out_soft2.mean(axis=0)
out_seg_mean_am2 = np.argmax(out_soft_mean2, axis=1)
out_seg_mean2    = torch.nn.functional.one_hot(torch.as_tensor(out_seg_mean_am2), num_classes=4).detach().cpu().numpy()


# OBS: PHASE
ref = torch.nn.functional.one_hot(torch.as_tensor(Tensor(gt_test_es_sub).to(torch.int64)), num_classes=4).detach().cpu().numpy()

#% Metrics
os.chdir("C:/Users/katrine/Documents/GitHub/Speciale2021")
#os.chdir("/Users/michalablicher/Documents/GitHub/Speciale2021")
from metrics import accuracy_self, EF_calculation, dc, hd, jc, precision, mcc, recall, risk, sensitivity, specificity, true_negative_rate, true_positive_rate, positive_predictive_value, hd95, assd, asd, ravd, volume_correlation, volume_change_correlation, obj_assd, obj_asd, obj_fpr, obj_tpr

dice1 = np.zeros((out_seg_mean1.shape[0],3))
dice2 = np.zeros((out_seg_mean2.shape[0],3))

haus95_1 = np.zeros((out_seg_mean1.shape[0],3))
haus95_2 = np.zeros((out_seg_mean2.shape[0],3))

for i in range(0,out_seg_mean1.shape[0]):
      
    dice1[i,0] = dc(out_seg_mean1[i,:,:,1],ref[i,:,:,1])  # = RV
    dice1[i,1] = dc(out_seg_mean1[i,:,:,2],ref[i,:,:,2])  # = MYO
    dice1[i,2] = dc(out_seg_mean1[i,:,:,3],ref[i,:,:,3])  # = LV
    
    dice2[i,0] = dc(out_seg_mean2[i,:,:,1],ref[i,:,:,1])  # = RV
    dice2[i,1] = dc(out_seg_mean2[i,:,:,2],ref[i,:,:,2])  # = MYO
    dice2[i,2] = dc(out_seg_mean2[i,:,:,3],ref[i,:,:,3])  # = LV
    
    # If there is no prediction or annotation then don't calculate Hausdorff distance and
    # skip to calculation for next class
    h_count = 0
    
    if len(np.unique(ref[i,:,:,1]))!=1 and len(np.unique(out_seg_mean1[i,:,:,1]))!=1 and len(np.unique(out_seg_mean2[i,:,:,1]))!=1:
        haus95_1[i,0]    = hd95(out_seg_mean1[i,:,:,1],ref[i,:,:,1])  
        haus95_2[i,0]    = hd95(out_seg_mean2[i,:,:,1],ref[i,:,:,1])  
        h_count += 1
    else:
        pass

    
    if len(np.unique(ref[i,:,:,2]))!=1 and len(np.unique(out_seg_mean1[i,:,:,2]))!=1 and len(np.unique(out_seg_mean2[i,:,:,2]))!=1:      
        haus95_1[i,1]    = hd95(out_seg_mean1[i,:,:,2],ref[i,:,:,2])
        haus95_2[i,1]    = hd95(out_seg_mean2[i,:,:,2],ref[i,:,:,2])
        h_count += 1
    else:
        pass
    
    if len(np.unique(ref[i,:,:,3]))!=1 and len(np.unique(out_seg_mean1[i,:,:,3]))!=1 and len(np.unique(out_seg_mean2[i,:,:,3]))!=1:
        haus95_1[i,2]    = hd95(out_seg_mean1[i,:,:,3],ref[i,:,:,3])
        haus95_2[i,2]    = hd95(out_seg_mean2[i,:,:,3],ref[i,:,:,3])
        h_count += 1
    else:
        pass
    
        pass        
    if h_count!= 3:
        print('Haus not calculated for all classes for slice: ', i)
    else:
        pass 
    
#%
from scipy.stats import mannwhitneyu
tissue = ['RV','MYO','LV']

c = 0

dice_m1 = dice1[:,c]
dice_m2 = dice2[:,c]

haus_m1 = haus95_1[:,c]
haus_m2 = haus95_2[:,c]

resd = mannwhitneyu(dice_m1, dice_m2, alternative="two-sided")
resh = mannwhitneyu(haus_m1, haus_m2, alternative="two-sided")

print('\n')
print('Dice: ',tissue[c],resd)
print('Haus: ',tissue[c],resh)
print('\n')
c = 1

dice_m1 = dice1[:,c]
dice_m2 = dice2[:,c]

haus_m1 = haus95_1[:,c]
haus_m2 = haus95_2[:,c]

resd = mannwhitneyu(dice_m1, dice_m2, alternative="two-sided")
resh = mannwhitneyu(haus_m1, haus_m2, alternative="two-sided")

print('Dice: ',tissue[c],resd)
print('Haus: ',tissue[c],resh)
print('\n')
c = 2

dice_m1 = dice1[:,c]
dice_m2 = dice2[:,c]

haus_m1 = haus95_1[:,c]
haus_m2 = haus95_2[:,c]

resd = mannwhitneyu(dice_m1, dice_m2, alternative="two-sided")
resh = mannwhitneyu(haus_m1, haus_m2, alternative="two-sided")

print('Dice: ',tissue[c],resd)
print('Haus: ',tissue[c],resh)
print('\n')














