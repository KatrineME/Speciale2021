#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 11:31:57 2021

@author: michalablicher
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 11:41:06 2021

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


if torch.cuda.is_available():
    # Tensor = torch.cuda.FloatTensor
    device = 'cuda'
else:
    # Tensor = torch.FloatTensor
    device = 'cpu'
torch.cuda.manual_seed_all(808)


#%% Specify directory
user = 'GPU'

if user == 'M':
    os.chdir('/Users/michalablicher/Documents/GitHub/Speciale2021')
if user == 'K':
    os.chdir('C:/Users/katrine/Documents/GitHub/Speciale2021')
if user == 'GPU':
    os.chdir('/home/katrine/Speciale2021/Speciale2021')
    
    
from load_data_gt_im_sub_space import load_data_sub

data_im_ed_DCM,  data_gt_ed_DCM  = load_data_sub(user,'Diastole','DCM')
data_im_ed_HCM,  data_gt_ed_HCM  = load_data_sub(user,'Diastole','HCM')
data_im_ed_MINF, data_gt_ed_MINF = load_data_sub(user,'Diastole','MINF')
data_im_ed_NOR,  data_gt_ed_NOR  = load_data_sub(user,'Diastole','NOR')
data_im_ed_RV,   data_gt_ed_RV   = load_data_sub(user,'Diastole','RV')

#%% BATCH GENERATOR
num_train_sub = 12
num_eval_sub  = num_train_sub

num_train_res = num_eval_sub + 6
num_test_res  = num_train_res + 2

im_train_es_res = np.concatenate((np.concatenate(data_im_ed_DCM[num_eval_sub:num_train_res]).astype(None),
                                  np.concatenate(data_im_ed_HCM[num_eval_sub:num_train_res]).astype(None),
                                  np.concatenate(data_im_ed_MINF[num_eval_sub:num_train_res]).astype(None),
                                  np.concatenate(data_im_ed_NOR[num_eval_sub:num_train_res]).astype(None),
                                  np.concatenate(data_im_ed_RV[num_eval_sub:num_train_res]).astype(None)))

gt_train_es_res = np.concatenate((np.concatenate(data_gt_ed_DCM[num_eval_sub:num_train_res]).astype(None),
                                  np.concatenate(data_gt_ed_HCM[num_eval_sub:num_train_res]).astype(None),
                                  np.concatenate(data_gt_ed_MINF[num_eval_sub:num_train_res]).astype(None),
                                  np.concatenate(data_gt_ed_NOR[num_eval_sub:num_train_res]).astype(None),
                                  np.concatenate(data_gt_ed_RV[num_eval_sub:num_train_res]).astype(None)))


im_test_es_res = np.concatenate((np.concatenate(data_im_ed_DCM[num_train_res:num_test_res]).astype(None),
                                  np.concatenate(data_im_ed_HCM[num_train_res:num_test_res]).astype(None),
                                  np.concatenate(data_im_ed_MINF[num_train_res:num_test_res]).astype(None),
                                  np.concatenate(data_im_ed_NOR[num_train_res:num_test_res]).astype(None),
                                  np.concatenate(data_im_ed_RV[num_train_res:num_test_res]).astype(None)))

gt_test_es_res = np.concatenate((np.concatenate(data_gt_ed_DCM[num_train_res:num_test_res]).astype(None),
                                  np.concatenate(data_gt_ed_HCM[num_train_res:num_test_res]).astype(None),
                                  np.concatenate(data_gt_ed_MINF[num_train_res:num_test_res]).astype(None),
                                  np.concatenate(data_gt_ed_NOR[num_train_res:num_test_res]).astype(None),
                                  np.concatenate(data_gt_ed_RV[num_train_res:num_test_res]).astype(None)))
print('Data loaded+concat')

#%% U-Net
# LOAD THE SOFTMAX PROBABILITES OF THE 6 FOLD MODELS
#% Load softmax from ensemble models
#PATH_softmax_ensemble_unet = 'C:/Users/katrine/Desktop/Optuna/Out_softmax_fold_avg_test_ResNet.pt'
PATH_softmax_ensemble_unet = '/home/michalablicher/Speciale2021/Speciale2021/Out_softmax_fold_avg_dice_lclv_dia_150e_opt_train_ResNet.pt'

#PATH_softmax_ensemble_unet = '/home/katrine/Speciale2021/Speciale2021/Out_softmax_fold_avg.pt'
out_softmax_unet_fold = torch.load(PATH_softmax_ensemble_unet ,  map_location=torch.device(device))

# mean them over dim=0
#out_softmax_unet = out_softmax_unet_fold[:,252:,:,:,:].mean(axis=0)
out_softmax_unet = out_softmax_unet_fold.mean(axis=0)

#Argmax
seg_met = np.argmax(out_softmax_unet, axis=1)

# One hot encode
seg_oh = torch.nn.functional.one_hot(torch.as_tensor(seg_met), num_classes=4).detach().cpu().numpy()
ref_oh = torch.nn.functional.one_hot(Tensor(gt_test_es_res).to(torch.int64), num_classes=4).detach().cpu().numpy()

#%% E-map
import scipy.stats
emap = np.zeros((seg_oh.shape[0],seg_oh.shape[1],seg_oh.shape[2]))

for i in range(0, emap.shape[0]):

    out_img = out_softmax_unet[i,:,:,:]#.detach().cpu().numpy())
    entropy2 = scipy.stats.entropy(out_img)
    
    # Normalize 
    m_entropy   = np.max(entropy2)
    entropy     = entropy2/m_entropy
    emap[i,:,:] = entropy

emap = np.expand_dims(emap, axis=1)
#%%
#% Wrap all inputs together
im     = Tensor(im_test_es_res)
umap   = Tensor(emap)
seg    = Tensor(np.expand_dims(seg_met, axis = 1))

print('Sizes of concat: im, umap, seg',im.shape,umap.shape,seg.shape)

input_concat = torch.cat((im,umap,seg), dim=1)

#%%
H = 16
W = 16
CV_folds = 6
data_dim = input_concat.shape[0]

out_patch = np.zeros((CV_folds, data_dim, 2, H, W))

input_data = torch.utils.data.DataLoader(input_concat, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, prefetch_factor=2, num_workers=0)


for fold in range(0,6):
    if user == 'GPU':
        path_model ='/home/michala/Speciale2021/Speciale2021/Trained_Detection_dice_lclv_dia_fold_150{}.pt'.format(fold)
    if user == 'K':
        path_model = 'C:/Users/katrine/Desktop/Optuna/Trained_Detection_CE_dia_fold_500{}.pt'.format(fold)
    if user == 'M':
        path_model = '/Users/michalablicher/Desktop/Trained_Detection_CE_dia_fold_500{}.pt'.format(fold)
    model = torch.load(path_model, map_location=torch.device(device))
    model.eval()
    
    for i, (im) in enumerate(input_data):
        im = Tensor.numpy(im)
        
        out = model(Tensor(im).cuda())
        #out = model(Tensor(im))
        out_patch[fold,i,:,:,:] = out["softmax"].detach().cpu().numpy() 
        
    del path_model, model, out
    print('Done for fold',fold)

if user == 'GPU':
    PATH_out_patch = '/home/michala/Speciale2021/Speciale2021/Out_patch_avg_dice_lclv_dia_fold_150.pt'
if user == 'K':
    PATH_out_patch = 'C:/Users/katrine/Desktop/Optuna/Out_patch_fold_500_avg.pt'
if user == 'M':
    PATH_out_patch = '/Users/michalablicher/Desktop/Out_patch_fold_500_avg.pt'
    
torch.save(out_patch, PATH_out_patch)










