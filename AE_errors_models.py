# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 09:20:49 2021

@author: katrine
"""
import torch
import os
import nibabel as nib
import numpy   as np
import pandas as pd
import torchvision
import glob2
import torch.optim as optim
from scipy import ndimage
from sklearn.model_selection import KFold
import seaborn as sns

from torch.autograd  import Variable
from torch import nn
from torch import Tensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    # Tensor = torch.cuda.FloatTensor
    device = 'cuda'
else:
    # Tensor = torch.FloatTensor
    device = 'cpu'
torch.cuda.manual_seed_all(808)

#%% Import results from training (Loss + Accuracy)
PATH_SD     = 'C:/Users/katrine/Desktop/Optuna/Final CV models/Out_softmax_fold_avg_150dia_dice.pt'
PATH_SD_opt = 'C:/Users/katrine/Desktop/Optuna/Final CV models/Out_softmax_fold_avg_150dia_dice_opt.pt'
PATH_AE     = 'C:/Users/katrine/Desktop/Optuna/Final CV models/Out_softmax_fold_avg_150dia_dice_lclv.pt'
PATH_AE_opt = 'C:/Users/katrine/Desktop/Optuna/Final CV models/Out_softmax_fold_avg_150dia_dice_lclv_opt.pt'

SD     = torch.load(PATH_SD, map_location=torch.device('cpu'))
SD_opt = torch.load(PATH_SD_opt, map_location=torch.device('cpu'))
AE     = torch.load(PATH_AE, map_location=torch.device('cpu'))
AE_opt = torch.load(PATH_AE_opt, map_location=torch.device('cpu'))
#%%
SD_mean     = SD.mean(axis=0)
SD_mean_am  = np.argmax(SD_mean, axis=1)
SD_seg_mean = torch.nn.functional.one_hot(torch.as_tensor(SD_mean_am), num_classes=4).detach().cpu().numpy()

SD_opt_mean     = SD_opt.mean(axis=0)
SD_opt_mean_am  = np.argmax(SD_opt_mean, axis=1)
SD_opt_seg_mean = torch.nn.functional.one_hot(torch.as_tensor(SD_opt_mean_am), num_classes=4).detach().cpu().numpy()

AE_mean     = AE.mean(axis=0)
AE_mean_am  = np.argmax(AE_mean, axis=1)
AE_seg_mean = torch.nn.functional.one_hot(torch.as_tensor(AE_mean_am), num_classes=4).detach().cpu().numpy()

AE_opt_mean     = AE_opt.mean(axis=0)
AE_opt_mean_am  = np.argmax(AE_opt_mean, axis=1)
AE_opt_seg_mean = torch.nn.functional.one_hot(torch.as_tensor(AE_opt_mean_am), num_classes=4).detach().cpu().numpy()

#%% #################################################################################################################
##################################################################################################################
os.chdir('C:/Users/katrine/Documents/GitHub/Speciale2021')
from load_data_gt_im_sub_space import load_data_sub
phase = 'Diastole'
user = 'K'

data_im_ed_DCM,  data_gt_ed_DCM  = load_data_sub(user,phase,'DCM')
data_im_ed_HCM,  data_gt_ed_HCM  = load_data_sub(user,phase,'HCM')
data_im_ed_MINF, data_gt_ed_MINF = load_data_sub(user,phase,'MINF')
data_im_ed_NOR,  data_gt_ed_NOR  = load_data_sub(user,phase,'NOR')
data_im_ed_RV,   data_gt_ed_RV   = load_data_sub(user,phase,'RV')

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

#%%
slice_1 = 34
slice_2 = 104
slice_3 = 264

#slice_1 = 264
#slice_2 = 271
#slice_3 = 315

alpha = 0.3

plt.figure(dpi=300, figsize=(10,17))
plt.subplot(5,3,1)
plt.imshow(gt_test_ed_sub[slice_1,:,:])
#plt.imshow(im_test_ed_sub[slice_1,0,:,:], alpha=alpha)
plt.ylabel('Reference', fontsize=20)
plt.title('Slice: {}'.format(slice_1), fontsize=20)
plt.subplot(5,3,2)
plt.imshow(gt_test_ed_sub[slice_2,:,:])
#plt.imshow(im_test_ed_sub[slice_2,0,:,:], alpha=alpha)
plt.title('Slice: {}'.format(slice_2), fontsize=20)
plt.subplot(5,3,3)
plt.imshow(gt_test_ed_sub[slice_3,:,:])
#plt.imshow(im_test_ed_sub[slice_3,0,:,:], alpha=alpha)
plt.title('Slice: {}'.format(slice_3), fontsize=20)

plt.subplot(5,3,4)
plt.imshow(SD_mean_am[slice_1,:,:])
#plt.imshow(im_test_ed_sub[slice_1,0,:,:], alpha=alpha)
plt.ylabel('SD model', fontsize=20)
plt.subplot(5,3,5)
plt.imshow(SD_mean_am[slice_2,:,:])
#plt.imshow(im_test_ed_sub[slice_2,0,:,:], alpha=alpha)
plt.subplot(5,3,6)
plt.imshow(SD_mean_am[slice_3,:,:])
#plt.imshow(im_test_ed_sub[slice_3,0,:,:], alpha=alpha)

plt.subplot(5,3,7)
plt.imshow(SD_opt_mean_am[slice_1,:,:])
#plt.imshow(im_test_ed_sub[slice_1,0,:,:], alpha=alpha)
plt.ylabel('SD opt model', fontsize=20)
plt.subplot(5,3,8)
plt.imshow(SD_opt_mean_am[slice_2,:,:])
#plt.imshow(im_test_ed_sub[slice_2,0,:,:], alpha=alpha)
plt.subplot(5,3,9)
plt.imshow(SD_opt_mean_am[slice_3,:,:])
#plt.imshow(im_test_ed_sub[slice_3,0,:,:], alpha=alpha)

plt.subplot(5,3,10)
plt.imshow(AE_mean_am[slice_1,:,:])
#plt.imshow(im_test_ed_sub[slice_1,0,:,:], alpha=alpha)
plt.ylabel('AE model', fontsize=20)
plt.subplot(5,3,11)
plt.imshow(AE_mean_am[slice_2,:,:])
#plt.imshow(im_test_ed_sub[slice_2,0,:,:], alpha=alpha)
plt.subplot(5,3,12)
plt.imshow(AE_mean_am[slice_3,:,:])
#plt.imshow(im_test_ed_sub[slice_3,0,:,:], alpha=alpha)

plt.subplot(5,3,13)
plt.imshow(AE_opt_mean_am[slice_1,:,:])
#plt.imshow(im_test_ed_sub[slice_1,0,:,:], alpha=alpha)
plt.ylabel('AE opt model', fontsize=20)
plt.subplot(5,3,14)
plt.imshow(AE_opt_mean_am[slice_2,:,:])
#plt.imshow(im_test_ed_sub[slice_2,0,:,:], alpha=alpha)
plt.subplot(5,3,15)
plt.imshow(AE_opt_mean_am[slice_3,:,:])
#plt.imshow(im_test_ed_sub[slice_3,0,:,:], alpha=alpha)

#%%
#slice_1 = 34
#slice_2 = 104
#slice_3 = 264

alpha = 0.3

plt.figure(dpi=300, figsize=(10,18))
plt.subplot(6,3,1)
plt.imshow(im_test_ed_sub[slice_1,0,:,:])
plt.ylabel('Original cMRI', fontsize=20)
plt.title('Slice: {}'.format(slice_1), fontsize=20)
plt.subplot(6,3,2)
plt.imshow(im_test_ed_sub[slice_2,0,:,:])
plt.title('Slice: {}'.format(slice_2), fontsize=20)
plt.subplot(6,3,3)
plt.imshow(im_test_ed_sub[slice_3,0,:,:])
plt.title('Slice: {}'.format(slice_3), fontsize=20)

plt.subplot(6,3,1+3)
plt.imshow(gt_test_ed_sub[slice_1,:,:])
plt.ylabel('Reference', fontsize=20)
plt.subplot(6,3,2+3)
plt.imshow(gt_test_ed_sub[slice_2,:,:])
plt.subplot(6,3,3+3)
plt.imshow(gt_test_ed_sub[slice_3,:,:])


plt.subplot(6,3,4+3)
plt.imshow(SD_mean_am[slice_1,:,:])
plt.ylabel('SD model', fontsize=20)
plt.subplot(6,3,5+3)
plt.imshow(SD_mean_am[slice_2,:,:])
plt.subplot(6,3,6+3)
plt.imshow(SD_mean_am[slice_3,:,:])

plt.subplot(6,3,7+3)
plt.imshow(SD_opt_mean_am[slice_1,:,:])
plt.ylabel('SD opt model', fontsize=20)
plt.subplot(6,3,8+3)
plt.imshow(SD_opt_mean_am[slice_2,:,:])
plt.subplot(6,3,9+3)
plt.imshow(SD_opt_mean_am[slice_3,:,:])

plt.subplot(6,3,10+3)
plt.imshow(AE_mean_am[slice_1,:,:])
plt.ylabel('AE model', fontsize=20)
plt.subplot(6,3,11+3)
plt.imshow(AE_mean_am[slice_2,:,:])
plt.subplot(6,3,12+3)
plt.imshow(AE_mean_am[slice_3,:,:])

plt.subplot(6,3,13+3)
plt.imshow(AE_opt_mean_am[slice_1,:,:])
plt.ylabel('AE opt model', fontsize=20)
plt.subplot(6,3,14+3)
plt.imshow(AE_opt_mean_am[slice_2,:,:])
plt.subplot(6,3,15+3)
plt.imshow(AE_opt_mean_am[slice_3,:,:])

#%%
alpha = 0.3

plt.figure(dpi=300, figsize=(20,10))
plt.subplot(3,6,1)
plt.imshow(im_test_ed_sub[slice_1,0,:,:])
plt.title('Original cMRI', fontsize=20)
plt.ylabel('Slice: {}'.format(slice_1), fontsize=20)
plt.subplot(3,6,1+6)
plt.imshow(im_test_ed_sub[slice_2,0,:,:])
plt.ylabel('Slice: {}'.format(slice_2), fontsize=20)
plt.subplot(3,6,1+12)
plt.imshow(im_test_ed_sub[slice_3,0,:,:])
plt.ylabel('Slice: {}'.format(slice_3), fontsize=20)

plt.subplot(3,6,2)
plt.imshow(gt_test_ed_sub[slice_1,:,:])
plt.title('Reference', fontsize=20)
plt.subplot(3,6,2+6)
plt.imshow(gt_test_ed_sub[slice_2,:,:])
plt.subplot(3,6,2+12)
plt.imshow(gt_test_ed_sub[slice_3,:,:])


plt.subplot(3,6,3)
plt.imshow(SD_mean_am[slice_1,:,:])
plt.title('SD model', fontsize=20)
plt.subplot(3,6,3+6)
plt.imshow(SD_mean_am[slice_2,:,:])
plt.subplot(3,6,3+12)
plt.imshow(SD_mean_am[slice_3,:,:])

plt.subplot(3,6,4)
plt.imshow(SD_opt_mean_am[slice_1,:,:])
plt.title('SD opt model', fontsize=20)
plt.subplot(3,6,4+6)
plt.imshow(SD_opt_mean_am[slice_2,:,:])
plt.subplot(3,6,4+12)
plt.imshow(SD_opt_mean_am[slice_3,:,:])

plt.subplot(3,6,5)
plt.imshow(AE_mean_am[slice_1,:,:])
plt.title('AE model', fontsize=20)
plt.subplot(3,6,5+6)
plt.imshow(AE_mean_am[slice_2,:,:])
plt.subplot(3,6,5+12)
plt.imshow(AE_mean_am[slice_3,:,:])

plt.subplot(3,6,6)
plt.imshow(AE_opt_mean_am[slice_1,:,:])
plt.title('AE opt model', fontsize=20)
plt.subplot(3,6,6+6)
plt.imshow(AE_opt_mean_am[slice_2,:,:])
plt.subplot(3,6,6+12)
plt.imshow(AE_opt_mean_am[slice_3,:,:])
