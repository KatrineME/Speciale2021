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
user = 'M'

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
PATH_softmax_ensemble_unet = '/Users/michalablicher/Desktop/Out_softmax_fold_avg_dice_lclv_dia_150e_opt_test_ResNet.pt'

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

image = 4*10

plt.figure(dpi=200, figsize = (15,15))

for i in range(0, 4):
    plt.subplot(4,4,1+i)
    #plt.subplots_adjust(wspace = 0.4)
    plt.imshow(im[i+image,0,:,:])
    plt.title('cMRI') 
    
    plt.subplot(4,4,5+i)
    plt.imshow(seg[i+image,0,:,:])
    plt.title('Segmentation') 
   
    plt.subplot(4,4,9+i)
    plt.imshow(umap[i+image,0,:,:])   
    plt.title('U-map') 
    
    plt.subplot(4,4,13+i)
    plt.imshow(gt_test_es_res[i+image,:,:])   
    plt.title('Reference') 

#%%
#%%

plt.figure(dpi=200, figsize = (15,15))
i = 0
image = 29
plt.subplot(4,1,1)
#plt.suptitle('Input for detection network', y=0.67, fontsize=25)
#plt.subplots_adjust(wspace = 0.4)
plt.imshow(im[i+image,0,:,:])
#plt.title('cMRI', fontsize=20) 

plt.subplot(4,1,2)
plt.imshow(umap[i+image,0,:,:])   
#plt.title('Segmentation', fontsize=20) 
   
plt.subplot(4,1,3)
plt.imshow(seg[i+image,0,:,:])
#plt.title('U-map', fontsize=20) 

plt.subplot(4,1,4)
plt.imshow(gt_test_es_res[i+image,:,:])   
#plt.title('U-map', fontsize=15) 

#%%
"""
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
        path_model ='/home/katrine/Speciale2021/Speciale2021/Trained_Detection_CE_dia_fold_500{}.pt'.format(fold)
    if user == 'K':
        path_model = 'C:/Users/katrine/Desktop/Optuna/Trained_Detection_CE_dia_fold_500{}.pt'.format(fold)
    if user == 'M':
        path_model = '/Users/michalablicher/Desktop/Trained_Detection_CE_dia_fold_500{}.pt'.format(fold)
    model = torch.load(path_model, map_location=torch.device(device))
    model.eval()
    
    for i, (im) in enumerate(input_data):
        im = Tensor.numpy(im)
        
        #out = model(Tensor(im).cuda())
        out = model(Tensor(im))
        out_patch[fold,i,:,:,:] = out["softmax"].detach().cpu().numpy() 
        
    del path_model, model, out
    print('Done for fold',fold)

if user == 'GPU':
    PATH_out_patch = '/home/katrine/Speciale2021/Speciale2021/Out_patch_fold_500_avg.pt'
if user == 'K':
    PATH_out_patch = 'C:/Users/katrine/Desktop/Optuna/Out_patch_fold_500_avg.pt'
if user == 'M':
    PATH_out_patch = '/Users/michalablicher/Desktop/Out_patch_fold_500_avg.pt'
    
torch.save(out_patch, PATH_out_patch)
"""
#%%

out_patch_load = '/Users/michalablicher/Desktop/Out_patch_avg_dice_lclv_dia_fold_150.pt'
out_patch_softmax_fold = torch.load(out_patch_load ,  map_location=torch.device(device))

#%% Plot
mean_patch = out_patch_softmax_fold.mean(axis=0)

m_patch = mean_patch > 0.15

size = 6
slice = 32
#%%
"""
plt.figure(dpi=200)
plt.subplot(2,4,1)
plt.imshow(out_patch[5,slice,1,:,:])
plt.title('Softmax patch fold 0',fontsize=size)
plt.colorbar()

plt.subplot(2,4,2)
plt.imshow(out_patch[1,slice,1,:,:])
plt.title('Softmax patch fold 1',fontsize=size)
plt.colorbar()

plt.subplot(2,4,3)
plt.imshow(out_patch[2,slice,1,:,:])
plt.title('Softmax patch fold 2',fontsize=size)
plt.colorbar()

plt.subplot(2,4,4)
plt.imshow(out_patch[3,slice,1,:,:])
plt.title('Softmax patch fold 3',fontsize=size)
plt.colorbar()

plt.subplot(2,4,5)
plt.imshow(out_patch[4,slice,1,:,:])
plt.title('Softmax patch fold 4',fontsize=size)
plt.colorbar()

plt.subplot(2,4,6)
plt.imshow(out_patch[5,slice,1,:,:])
plt.title('Softmax patch fold 5',fontsize=size)
plt.colorbar()
"""
#%%
slice = 3

plt.subplot(1,2,1)
plt.imshow(mean_patch[slice,1,:,:])
plt.title('Mean softmax patch',fontsize=size)
plt.colorbar()

plt.subplot(1,2,2)
plt.imshow(m_patch[slice,1,:,:])
plt.title('Binarized at {}'.format('threshold'),fontsize=size)
plt.colorbar()

#%% Metrics on slices with failures

failure_per_slice = np.sum(m_patch[:,1,:,:], axis=(1,2))
failures = (np.count_nonzero(failure_per_slice)/failure_per_slice.shape[0])*100
print('Failures in {} % of test slices'.format(failures))

p = []

p.append(data_gt_ed_DCM[num_train_res:num_test_res][0].shape[0])
p.append(data_gt_ed_DCM[num_train_res:num_test_res][1].shape[0])


p.append(data_gt_ed_HCM[num_train_res:num_test_res][0].shape[0])
p.append(data_gt_ed_HCM[num_train_res:num_test_res][1].shape[0])
p.append(data_gt_ed_MINF[num_train_res:num_test_res][0].shape[0])
p.append(data_gt_ed_MINF[num_train_res:num_test_res][1].shape[0])


p.append(data_gt_ed_NOR[num_train_res:num_test_res][0].shape[0])
p.append(data_gt_ed_NOR[num_train_res:num_test_res][1].shape[0])

p.append(data_gt_ed_RV[num_train_res:num_test_res][0].shape[0])
p.append(data_gt_ed_RV[num_train_res:num_test_res][1].shape[0])  

#failure_per_patient = 
    

#%% Upsample

#% Upsample
image = 77
upper_image = image - 1
lower_image = image + 1


test_im = Tensor(np.expand_dims(m_patch[upper_image:lower_image,1,:,:],axis=0))

up = nn.Upsample((128,128), mode='bilinear', align_corners=True)
up_im = up(test_im) > 0


difference = (gt_test_es_res[image,:,:]-seg[image,0,:,:].detach().numpy())

plt.figure(dpi=200)
plt.subplot(1,4,1)
plt.subplots_adjust(wspace = 0.4)
plt.imshow(input_concat[image,2,:,:])
#plt.imshow(up_im[0,0,:,:])
#plt.imshow(input_concat[image,0,:,:], alpha= 0.4)
plt.title('Segmentation',fontsize=size)

plt.subplot(1,4,2)
plt.imshow(up_im[0,1,:,:])
plt.imshow(input_concat[image,0,:,:], alpha= 0.4)
plt.title('Error patch',fontsize=size)

plt.subplot(1,4,3)
plt.imshow(difference)
plt.title('Difference between ref-seg',fontsize=size)

plt.subplot(1,4,4)
plt.imshow(up_im[0,1,:,:])
plt.imshow(np.argmax((ref_oh[image,:,:,:]),axis=2), alpha= 0.6)
plt.imshow(input_concat[image,0,:,:], alpha= 0.4)
plt.title('Reference w. error',fontsize=size)

