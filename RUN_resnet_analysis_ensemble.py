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
#%%
H = 128
W = 128
CV_folds = 6
data_im = im_train_es_res.shape[0]


out_patch = np.zeros((CV_folds, data_im, 2, H, W))

im_data = torch.utils.data.DataLoader(im_train_es_res, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, prefetch_factor=2, num_workers=0)

for fold in range(0,6):
    if user == 'GPU':
        path_model ='/home/katrine/Speciale2021/Speciale2021/Trained_Detection_CE_dia_fold{}.pt'.format(fold)
    if user == 'K':
        path_model = 'C:/Users/katrine/Desktop/Optuna/Trained_Detection_CE_dia_fold{}.pt'.format(fold)
    model = torch.load(path_model, map_location=torch.device(device))
    model.eval()
    
    for i, (im) in enumerate(im_data):
        im = Tensor.numpy(im)
        
        out = model(Tensor(im).cuda())
        out_patch[fold,i,:,:,:] = out["softmax"].detach().cpu().numpy() 
        
    del path_model, model, out
    print('Done for fold',fold)

if user == 'GPU':
    PATH_out_patch = '/home/katrine/Speciale2021/Speciale2021/Out_patch_fold_avg.pt'
if user == 'K':
    PATH_out_patch = 'C:/Users/katrine/Desktop/Optuna/Out_patch_fold_avg.pt'
torch.save(out_patch, PATH_out_patch)

"""
#%% Run model0
path_model_0 = 'C:/Users/katrine/Desktop/Optuna/Trained_Unet_CE_dia_fold0.pt'
model_0 = torch.load(path_model_0, map_location=torch.device('cpu'))

model_0.eval()
out_0 = model_0(Tensor(im_test_ed_sub))
out_0 = out_0["softmax"]

#%% Run 
path_model_1 = 'C:/Users/katrine/Desktop/Optuna/Trained_Unet_CE_dia_fold1.pt'
model_1 = torch.load(path_model_1, map_location=torch.device('cpu'))

model_1.eval()
out_1 = model_1(Tensor(im_test_ed_sub))
out_1 = out_1["softmax"].detach().numpy()

#%% Run model2
path_model_2 = 'C:/Users/katrine/Desktop/Optuna/Trained_Unet_CE_dia_fold2.pt'
model_2 = torch.load(path_model_2, map_location=torch.device('cpu'))

model_2.eval()
out_2 = model_2(Tensor(im_test_ed_sub))
out_2 = out_2["softmax"].detach().numpy()

#%% Run model3
path_model_3 = 'C:/Users/katrine/Desktop/Optuna/Trained_Unet_CE_dia_fold3.pt'
model_3 = torch.load(path_model_3, map_location=torch.device('cpu'))

model_3.eval()
out_3 = model_3(Tensor(im_test_ed_sub))
out_3 = out_3["softmax"].detach().numpy()

#%% Run model4
path_model_4 = 'C:/Users/katrine/Desktop/Optuna/Trained_Unet_CE_dia_fold4.pt'
model_4 = torch.load(path_model_4, map_location=torch.device('cpu'))

model_4.eval()
out_4 = model_4(Tensor(im_test_ed_sub))
out_4 = out_4["softmax"].detach().numpy()

#%% Run model5
path_model_5 = 'C:/Users/katrine/Desktop/Optuna/Trained_Unet_CE_dia_fold5.pt'
model_5 = torch.load(path_model_5, map_location=torch.device('cpu'))

model_5.eval()
out_5 = model_5(Tensor(im_test_ed_sub))
out_5 = out_5["softmax"].detach().numpy()
"""

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
