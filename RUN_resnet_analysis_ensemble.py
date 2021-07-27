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

phase = 'Systole'
data_im_ed_DCM,  data_gt_ed_DCM  = load_data_sub(user,phase,'DCM')
data_im_ed_HCM,  data_gt_ed_HCM  = load_data_sub(user,phase,'HCM')
data_im_ed_MINF, data_gt_ed_MINF = load_data_sub(user,phase,'MINF')
data_im_ed_NOR,  data_gt_ed_NOR  = load_data_sub(user,phase,'NOR')
data_im_ed_RV,   data_gt_ed_RV   = load_data_sub(user,phase,'RV')

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
PATH_softmax_ensemble_unet = '/Users/michalablicher/Desktop/Out_softmax_fold_avg_dice_lclv_sys_150e_test_ResNet.pt'
#PATH_softmax_ensemble_unet = '/Users/michalablicher/Desktop/Out_softmax_fold_avg_dice_dia_150e_opt_train_ResNet.pt'

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

#ref_oh = torch.nn.functional.one_hot(Tensor(gt_train_es_res).to(torch.int64), num_classes=4).detach().cpu().numpy()
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

#im     = Tensor(im_train_es_res)

im     = Tensor(im_test_es_res)
umap   = Tensor(emap)
seg    = Tensor(np.expand_dims(seg_met, axis = 1))

print('Sizes of concat: im, umap, seg',im.shape,umap.shape,seg.shape)

input_concat = torch.cat((im, umap, seg), dim=1)

image = 4*2

plt.figure(dpi=200, figsize = (15,15))

for i in range(0, 4):
    plt.subplot(4,4,1+i)
    #plt.subplots_adjust(wspace = 0.4)
    plt.imshow(im[i+image,0,:,:])
    plt.title('cMRI slice {}'.format(i+image)) 
    
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
os.chdir("/Users/michalablicher/Documents/GitHub/Speciale2021")
from metrics import accuracy_self, EF_calculation, dc, hd, jc, precision, mcc, recall, risk, sensitivity, specificity, true_negative_rate, true_positive_rate, positive_predictive_value, hd95, assd, asd, ravd, volume_correlation, volume_change_correlation, obj_assd, obj_asd, obj_fpr, obj_tpr

umap_bin = umap > 0.5
umap_bin_s = np.squeeze(umap_bin)

#difference = (gt_train_es_res[:,:,:]-seg[:,0,:,:].detach().numpy())

difference = (gt_test_es_res[:,:,:]-seg[:,0,:,:].detach().numpy())
db = (difference != 0)
db_b = (db.astype(np.float))

dice_umap = np.zeros(85)

for i in range(0,umap_bin.shape[0]):
    dice_umap[i] = dc(umap_bin_s[i,:,:].detach().numpy() ,db_b[i,:,:]) 

difference_u = (umap_bin_s[:,:,:].detach().numpy()-db_b[:,:,:])

print('min dice', np.min(dice_umap))
print('max dice', np.max(dice_umap))
print('mean dice', np.mean(dice_umap))
print('var dice', np.var(dice_umap))


#%%
out_patch_load = '/Users/michalablicher/Desktop/Out_patch_avg_dice_sdloss_sys_fold_150.pt'


PATH_SI_dice_85 = '/Users/michalablicher/Desktop/SI_Tj_85_dice_lclv_sys.pt'
out_patch_softmax_fold = torch.load(out_patch_load ,  map_location=torch.device(device))
SI_set_85= torch.load(PATH_SI_dice_85 ,  map_location=torch.device(device))

#%% Plot
#mean_patch = out_patch_softmax_fold.mean(axis=0)

mean_patch = out_patch_softmax_fold[:,:,:,:,:]
m_patch_am = np.argmax(mean_patch, axis=1)


m_patch = mean_patch < 0.8
#%%
size = 5
slice = 5
plt.figure(dpi=200, figsize = (18,4))
for i in range(0,6):
    plt.subplot(2,6,1+i)
    plt.imshow(mean_patch[i,slice,1,:,:])
    plt.title('Mean softmax patch',fontsize=size)
    plt.colorbar(fraction=0.045)
    
    plt.subplot(2,6,7+i)
    plt.imshow(m_patch[i,slice,1,:,:])
    plt.title('Binarized at {}'.format('threshold'),fontsize=size)
    #plt.colorbar(fraction=0.045)


#%% Metrics on slices with failures

#failure_per_slice = np.sum(m_patch_am[:,:,:], axis=(1,2))
failure_per_slice = np.sum(m_patch[:,1,:,:], axis=(1,2))
failure_per_slice_tj = np.sum(SI_set_85[:,:,:], axis=(1,2))


failures = (np.count_nonzero(failure_per_slice)/failure_per_slice.shape[0])*100
print('Failures in {} % of test slices'.format(failures))

failures_tj = (np.count_nonzero(failure_per_slice_tj)/failure_per_slice_tj.shape[0])*100
print('Failures in tj {} % of test slices'.format(failures_tj))



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
mean_patch = out_patch_softmax_fold.mean(axis=0)
m_patch = mean_patch < 0.8

size = 20
slice = 15
plt.figure(dpi=200, figsize = (12,6))

plt.subplot(1,2,1)
plt.imshow(mean_patch[slice,0,:,:])
plt.title('Mean softmax patch',fontsize=size)
plt.colorbar(fraction=0.045)

plt.subplot(1,2,2)
plt.imshow(m_patch[slice,1,:,:])
plt.title('Binarized at {}'.format('threshold'),fontsize=size)
plt.colorbar(fraction=0.045)


#%%

#% Upsample
image = 14

s = np.expand_dims(SI_set_85[:,:,:],axis = 1)
test_im = Tensor(np.expand_dims(m_patch[image,:,:,:],axis=0))
lab_im = Tensor(np.expand_dims(s[image,:,:,:],axis=0))

up = nn.Upsample((128,128), mode='bilinear', align_corners=True)

up_im = up(test_im) > 0

up_lab = up(lab_im) > 0


size = 16
#difference = (gt_train_es_res[image,:,:]-seg[image,0,:,:].detach().numpy())

difference = (gt_test_es_res[image,:,:]-seg[image,0,:,:].detach().numpy())
db = np.array(difference != 0)

plt.figure(dpi=200, figsize =(12,8))
plt.subplot(1,3,2)
plt.suptitle('Slice {}'.format(image), y = 0.75, fontsize = 20)
#plt.subplots_adjust(wspace = 0.4)
plt.imshow(up_im[0,1,:,:])
plt.imshow(input_concat[image,2,:,:], alpha = 0.5)
#plt.imshow(up_im[0,0,:,:])
#plt.imshow(input_concat[image,0,:,:], alpha= 0.4)
plt.title('Error segmentation',fontsize=size)

plt.subplot(1,3,1)
plt.imshow(up_lab[0,0,:,:])
plt.imshow(np.argmax((ref_oh[image,:,:,:]),axis=2), alpha =0.6)
#plt.imshow(input_concat[image,2,:,:], alpha = 0.5)

#plt.imshow(input_concat[image,0,:,:], alpha= 0.4)
plt.title('Error reference label',fontsize=size)

plt.subplot(1,3,3)
plt.imshow(up_im[0,1,:,:])
plt.imshow(db, cmap='coolwarm', alpha = 0.5)
plt.title('Difference',fontsize=size)
"""
plt.subplot(1,4,4)
#plt.imshow(up_im[0,1,:,:])
plt.imshow(umap[image,0,:,:]) 
#plt.imshow(input_concat[image,0,:,:], alpha= 0.6)
plt.title('cMRI',fontsize=
"""


#%% Metrics
up_im = np.zeros((85,2,128,128))

up_lab = np.zeros((85,2,128,128))


s = np.expand_dims(SI_set_85[:,:,:],axis = 1)

for i in range(1,85):
    #% Upsample
    
    test_im = Tensor(np.expand_dims(m_patch[i,:,:,:],axis=0))
    lab_im = Tensor(np.expand_dims(s[i,:,:,:],axis=0))

    up = nn.Upsample((128,128), mode='bilinear', align_corners=True)
    up_im[i,:,:,:] = up(test_im) > 0
    up_lab[i,:,:,:] = up(lab_im) > 0


size = 16
difference = (gt_test_es_res[:,:,:]-seg[:,0,:,:].detach().numpy())
db = np.array(difference != 0)    


dice_error = np.zeros((85))
dice_16_error = np.zeros((85))

sen_error = np.zeros((85))
for i in range(0,85):
    dice_error[i] = dc(np.array(up_im[i,1,:,:]),np.array(up_lab[i,1,:,:]))
    sen_error[i] = sensitivity(np.array(up_im[i,1,:,:]),np.array(up_lab[i,1,:,:]))


print(np.mean(dice_error))

#%%


tpr = dc(np.array(up_im[:,1,:,:]),np.array(up_lab[:,1,:,:]))

tpr16 = dc(np.array(up_im[:,1,:,:]),np.array(up_lab[:,1,:,:]))


print(tpr)






