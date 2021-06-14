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
user = 'K'

if user == 'M':
    os.chdir('/Users/michalablicher/Documents/GitHub/Speciale2021')
if user == 'K':
    os.chdir('C:/Users/katrine/Documents/GitHub/Speciale2021')
if user == 'GPU':
    os.chdir('/home/katrine/Speciale2021/Speciale2021')
    
    
from load_data_gt_im_sub import load_data_sub

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
PATH_softmax_ensemble_unet = 'C:/Users/katrine/Desktop/Optuna/Out_softmax_fold_avg_test_ResNet.pt'
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

image = 2

plt.figure(dpi=200)
plt.subplot(1,4,1)
plt.subplots_adjust(wspace = 0.4)
plt.imshow(im[image,0,:,:])
plt.title('cMRI') 
plt.subplot(1,4,2)
plt.imshow(seg[image,0,:,:])
plt.title('Segmentation') 
plt.subplot(1,4,3)
plt.imshow(umap[image,0,:,:])   
plt.title('U-map') 
plt.subplot(1,4,4)
plt.imshow(gt_test_es_res[image,:,:])   
plt.title('ref') 

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
        path_model ='/home/katrine/Speciale2021/Speciale2021/Trained_Detection_CE_dia_fold{}.pt'.format(fold)
    if user == 'K':
        path_model = 'C:/Users/katrine/Desktop/Optuna/Trained_Detection_CE_dia_fold_300{}.pt'.format(fold)
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
    PATH_out_patch = '/home/katrine/Speciale2021/Speciale2021/Out_patch_fold_avg.pt'
if user == 'K':
    PATH_out_patch = 'C:/Users/katrine/Desktop/Optuna/Out_patch_fold_300_avg.pt'
torch.save(out_patch, PATH_out_patch)

#%% Plot

plt.imshow(out_patch[5,35,1,:,:])
plt.colorbar()

#%%
mean_patch = out_patch.mean(axis=0)

plt.imshow(mean_patch[35,1,:,:])
plt.colorbar()
#%%
m_patch = mean_patch>0.8

plt.imshow(m_patch[35,1,:,:])

#%% Upsample
k = m_patch

#% Upsample
image = 2
upper_image = image - 1
lower_image = image + 1

#test_im = Tensor(np.expand_dims(output_test[lower_image:upper_image,1,:,:],axis=0))
#test_im = Tensor(np.expand_dims(output_test[upper_image:lower_image,1,:,:],axis=0))
test_im = Tensor(np.expand_dims(k[upper_image:lower_image,1,:,:],axis=0))

up = nn.Upsample((128,128), mode='bilinear', align_corners=True)
up_im = up(test_im) > 0


difference = (gt_test_es_res[image,:,:]-seg[image,0,:,:].detach().numpy())

plt.figure(dpi=200)
plt.subplot(1,4,1)
plt.subplots_adjust(wspace = 0.4)
plt.imshow(input_concat[image,2,:,:])
#plt.imshow(up_im[0,0,:,:])
plt.imshow(input_concat[image,0,:,:], alpha= 0.4)
plt.title('Segmentation')

plt.subplot(1,4,2)
plt.imshow(up_im[0,1,:,:])
plt.imshow(input_concat[image,0,:,:], alpha= 0.4)
plt.title('Error patch')

plt.subplot(1,4,3)
plt.imshow(difference)
plt.title('Difference between ref+seg')

plt.subplot(1,4,3)
plt.imshow(up_im[0,1,:,:])
plt.imshow(np.argmax((ref_oh[image,:,:,:]),axis=2), alpha= 0.6)
plt.imshow(input_concat[image,0,:,:], alpha= 0.4)
plt.title('Reference w. error')




#%%
#Plot softmax probabilities for a single slice
test_slice = 300
alpha = 0.4

fig = plt.figure()

class_title = ['Background','Right Ventricle','Myocardium','Left Ventricle']
ref_dia = torch.nn.functional.one_hot(Tensor(gt_test_ed_sub).to(torch.int64), num_classes=4)
plt.figure(dpi=200, figsize=(18,32))

w = 0.1

for fold_model in range (0,6):
    out_img_ed = np.squeeze(out_soft[fold_model,test_slice,:,:,:])
    seg_met_dia = np.argmax(out_soft[fold_model,:,:,:], axis=1)
    seg_dia = torch.nn.functional.one_hot(torch.as_tensor(seg_met_dia), num_classes=4)
    
    #Reference annotation
    plt.suptitle('Diastolic phase: test image at slice %i for CV folds' %test_slice, fontsize=30, y=0.9)
    plt.subplot(7, 4, 1)
    plt.subplots_adjust(hspace = 0.05, wspace = w)
    plt.imshow(ref_dia[test_slice,:,:,0])
    plt.imshow(im_test_ed_sub[test_slice,0,:,:],alpha=alpha)
    plt.ylabel('Reference', fontsize=16)
    plt.title('Background', fontsize=16)
    
    plt.subplot(7, 4, 2)
    plt.subplots_adjust(hspace = 0.05, wspace = w)
    plt.imshow(ref_dia[test_slice,:,:,1])
    plt.imshow(im_test_ed_sub[test_slice,0,:,:],alpha=alpha)
    plt.title('Right ventricle', fontsize=16)
    
    plt.subplot(7, 4, 3)
    plt.subplots_adjust(hspace = 0.05, wspace = w)
    plt.imshow(ref_dia[test_slice,:,:,2])
    plt.imshow(im_test_ed_sub[test_slice,0,:,:],alpha=alpha)
    plt.title('Myocardium', fontsize=16)
    
    plt.subplot(7, 4, 4)
    plt.subplots_adjust(hspace = 0.05, wspace = w)
    plt.imshow(ref_dia[test_slice,:,:,3])
    plt.imshow(im_test_ed_sub[test_slice,0,:,:],alpha=alpha)
    plt.title('Left ventricle', fontsize=16)
    
    
    #CV model segmentations
    plt.subplot(7, 4, 1+4*(fold_model+1))
    plt.subplots_adjust(hspace = 0.05, wspace = w)
    plt.imshow(seg_dia[test_slice,:,:,0])
    plt.imshow(im_test_ed_sub[test_slice,0,:,:],alpha=alpha)
    plt.ylabel('CV fold {}'.format(fold_model), fontsize=16)
    
    plt.subplot(7, 4, 2+4*(fold_model+1))
    plt.subplots_adjust(hspace = 0.05, wspace = w)
    plt.imshow(seg_dia[test_slice,:,:,1])
    plt.imshow(im_test_ed_sub[test_slice,0,:,:],alpha=alpha)
    
    plt.subplot(7, 4, 3+4*(fold_model+1))
    plt.subplots_adjust(hspace = 0.05, wspace = w)
    plt.imshow(seg_dia[test_slice,:,:,2])
    plt.imshow(im_test_ed_sub[test_slice,0,:,:],alpha=alpha)
    
    plt.subplot(7, 4, 4+4*(fold_model+1))
    plt.subplots_adjust(hspace = 0.05, wspace = w)
    plt.imshow(seg_dia[test_slice,:,:,3])
    plt.imshow(im_test_ed_sub[test_slice,0,:,:],alpha=alpha)
        
    
    """
    plt.imshow(seg_dia[test_slice,:,:,i])
    plt.imshow(im_test_ed_sub[test_slice,0,:,:],alpha=alpha)
    plt.title(class_title[i], fontsize =16)
    plt.xticks(rotation=40, fontweight='light', fontsize=7)
    plt.yticks(horizontalalignment='right',fontweight='light',fontsize=7)
   
    if i == 0:
        plt.ylabel('Argmax fold {}'.format(fold_model), fontsize=14)
        

    plt.subplot(7, 4, i+1+4)     
    plt.subplots_adjust(hspace = 0.05, wspace = 0.2)
    plt.imshow(ref_dia[test_slice,:,:,i])
    plt.imshow(im_test_ed_sub[test_slice,0,:,:],alpha=alpha)
    if i == 0:
        plt.ylabel('Reference', fontsize=14)
        
    """
plt.show()  
