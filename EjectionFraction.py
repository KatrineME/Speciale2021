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
    # Tensor = torch.FloatTensory
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

 #%%
user = 'K'
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
path_soft_dia = 'C:/Users/katrine/Desktop/Optuna/Final CV models/Out_softmax_fold_avg_150dia_dice.pt'
path_soft_sys = 'C:/Users/katrine/Desktop/Optuna/Final CV models/Out_softmax_fold_avg_150sys_dice.pt'

#path_soft_dia = '/Users/michalablicher/Desktop/Out_softmax_fold_avg_150dia_CE.pt'
#path_soft_sys = '/Users/michalablicher/Desktop/Out_softmax_fold_avg_150dia_dice.pt'

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

#%%

slice = 27

plt.subplot(1,2,1)
plt.imshow(soft_dia_mean[slice,1,:,:])

plt.subplot(1,2,2)
plt.imshow(soft_sys_mean[slice,1,:,:])





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

EF_ref_RV    = EF_calculation(ref_vol_sys_RV, ref_vol_dia_RV, spacings)
EF_target_RV = EF_calculation(target_vol_sys_RV, target_vol_dia_RV, spacings)


ef_m_ref = np.mean(EF_ref[0])
ef_m_tar = np.mean(EF_target[0])

print('EF ref = \n',EF_ref[0]) 
print('EF seg = \n',EF_target[0]) 

print('End-sys volume ref = \n', EF_ref[1]) 
print('End-dia volume ref = \n', EF_ref[2]) 

print('End-sys volume seg = \n', EF_target[1]) 
print('End-dia volume seg = \n', EF_target[2]) 

#%% Correlation
cor_EF = np.corrcoef(EF_target[0], EF_ref[0])

cor_dia = np.corrcoef(target_vol_dia, ref_vol_dia)
cor_sys = np.corrcoef(target_vol_sys, ref_vol_sys)

cor_dia_RV = np.corrcoef(target_vol_dia_RV, ref_vol_dia_RV)
cor_sys_RV = np.corrcoef(target_vol_sys_RV, ref_vol_sys_RV)

#EF
print('Correlation EF LV =', cor_EF[1,0]) 
# LV
print('Correlation diastole LV vol =', cor_dia[1,0]) 
print('Correlation systole  LV vol =', cor_sys[1,0]) 
print('\n')
# RV
print('Correlation diastole RV vol =', cor_dia_RV[1,0]) 
print('Correlation systole  RV vol =', cor_sys_RV[1,0]) 


#%% Bias

bias_sys_vol = np.mean(EF_target[1]-EF_ref[1])
bias_dia_vol = np.mean(EF_target[2]-EF_ref[2])
bias_EF      = np.mean(EF_target[0]-EF_ref[0])

print('Bias sys LV=', bias_sys_vol ) 
print('Bias dia LV=', bias_dia_vol )
print('Bias EF  LV=', bias_EF )
print('\n')
bias_sys_vol_RV = np.mean(EF_target_RV[1]-EF_ref_RV[1])
bias_dia_vol_RV = np.mean(EF_target_RV[2]-EF_ref_RV[2])
bias_EF_RV      = np.mean(EF_target_RV[0]-EF_ref_RV[0])

print('Bias sys RV=', bias_sys_vol_RV ) 
print('Bias dia RV=', bias_dia_vol_RV )
print('Bias EF  RV=', bias_EF_RV )

#%% std

std_EF_dif  = np.std(EF_target[0]-EF_ref[0])
std_sys_dif = np.std(EF_target[1]-EF_ref[1])
std_dia_dif = np.std(EF_target[2]-EF_ref[2])

print('Std EF =', std_EF_dif )
print('Std sys vol =', std_sys_dif )
print('Std dia vol =', std_dia_dif ) 
print('\n')

std_EF_dif_RV = np.std(EF_target_RV[0]-EF_ref_RV[0])
std_sys_dif_RV = np.std(EF_target_RV[1]-EF_ref_RV[1])
std_dia_dif_RV = np.std(EF_target_RV[2]-EF_ref_RV[2])

print('Std EF RV =', std_EF_dif_RV )
print('Std sys vol RV =', std_sys_dif_RV )
print('Std dia vol RV =', std_dia_dif_RV ) 

#%% MAE

# LV
MAE_sys_vol = np.sum(np.abs(EF_target[1]-EF_ref[1]))/EF_ref[1].shape[0]
MAE_dia_vol = np.sum(np.abs(EF_target[2]-EF_ref[2]))/EF_ref[2].shape[0]
MAE_EF      = np.sum(np.abs(EF_ref[0]-EF_target[0]))/EF_ref[0].shape[0]

print('MAE EF =', MAE_EF )
print('MAE sys vol =', MAE_sys_vol )
print('MAE dia vol =', MAE_dia_vol ) 
print('\n')
# RV
MAE_sys_vol_RV = np.sum(np.abs(EF_target_RV[1]-EF_ref_RV[1]))/EF_ref_RV[1].shape[0]
MAE_dia_vol_RV = np.sum(np.abs(EF_target_RV[2]-EF_ref_RV[2]))/EF_ref_RV[2].shape[0]
MAE_EF_RV      = np.sum(np.abs(EF_target_RV[0]-EF_ref_RV[0]))/EF_ref_RV[0].shape[0]

print('MAE EF RV =', MAE_EF_RV )
print('MAE sys vol RV =', MAE_sys_vol_RV )
print('MAE dia vol RV =', MAE_dia_vol_RV ) 

#%% DICE SCORE
#%% Metrics
#os.chdir("C:/Users/katrine/Documents/GitHub/Speciale2021")
os.chdir('/Users/michalablicher/Documents/GitHub/Speciale2021')
from metrics import accuracy_self, EF_calculation, dc, hd, jc, precision, mcc, recall, risk, sensitivity, specificity, true_negative_rate, true_positive_rate, positive_predictive_value, hd95, assd, asd, ravd, volume_correlation, volume_change_correlation, obj_assd, obj_asd, obj_fpr, obj_tpr

dice_sys = np.zeros((out_seg_sys_mean.shape[0],3))

for i in range(0,out_seg_sys_mean.shape[0]):
    dice_sys[i,0] = dc(out_seg_sys_mean[i,:,:,1],ref_sys[i,:,:,1])  # = RV
    dice_sys[i,1] = dc(out_seg_sys_mean[i,:,:,2],ref_sys[i,:,:,2])  # = MYO
    dice_sys[i,2] = dc(out_seg_sys_mean[i,:,:,3],ref_sys[i,:,:,3])  # = LV
    
dice_dia = np.zeros((out_seg_dia_mean.shape[0],3))

for i in range(0,out_seg_sys_mean.shape[0]):
    dice_dia[i,0] = dc(out_seg_dia_mean[i,:,:,1],ref_sys[i,:,:,1])  # = RV
    dice_dia[i,1] = dc(out_seg_dia_mean[i,:,:,2],ref_sys[i,:,:,2])  # = MYO
    dice_dia[i,2] = dc(out_seg_dia_mean[i,:,:,3],ref_sys[i,:,:,3])  # = LV
    

#%% DICE SCORE
#%% Metrics
#os.chdir("C:/Users/katrine/Documents/GitHub/Speciale2021")
os.chdir('/Users/michalablicher/Documents/GitHub/Speciale2021')
from metrics import accuracy_self, EF_calculation, dc, hd, jc, precision, mcc, recall, risk, sensitivity, specificity, true_negative_rate, true_positive_rate, positive_predictive_value, hd95, assd, asd, ravd, volume_correlation, volume_change_correlation, obj_assd, obj_asd, obj_fpr, obj_tpr

dice_sys = np.zeros((out_seg_sys_mean.shape[0],3))

for i in range(0,out_seg_sys_mean.shape[0]):
    dice_sys[i,0] = dc(out_seg_sys_mean[i,:,:,1],ref_dia[i,:,:,1])  # = RV
    dice_sys[i,1] = dc(out_seg_sys_mean[i,:,:,2],ref_dia[i,:,:,2])  # = MYO
    dice_sys[i,2] = dc(out_seg_sys_mean[i,:,:,3],ref_dia[i,:,:,3])  # = LV
    
dice_dia = np.zeros((out_seg_dia_mean.shape[0],3))

for i in range(0,out_seg_sys_mean.shape[0]):
    dice_dia[i,0] = dc(out_seg_dia_mean[i,:,:,1],ref_dia[i,:,:,1])  # = RV
    dice_dia[i,1] = dc(out_seg_dia_mean[i,:,:,2],ref_dia[i,:,:,2])  # = MYO
    dice_dia[i,2] = dc(out_seg_dia_mean[i,:,:,3],ref_dia[i,:,:,3])  # = LV
    
#%% Histogram
plt.figure(dpi=300)
diff_dia = EF_ref[2]-EF_target[2]
plt.hist(np.abs(diff_dia),bins=40)
plt.title('Histogram of absolute difference in volume (dia)')

#%% Histogram
plt.figure(dpi=200, figsize =(12,12))
plt.subplot(2,1,1)
plt.hist(dice_sys, label=('RV','MYO','LV'), color=('tab:blue','mediumseagreen','orange'))
plt.legend()
plt.xlabel('Dice Score')
plt.ylabel('# Number of slices')
plt.title('Histogram of Dice Scores for sys model')
plt.grid(True, color = "grey", linewidth = "0.5", linestyle = "-")


plt.subplot(2,1,2)
plt.hist(dice_dia, label=('RV','MYO','LV'), color=('tab:blue','mediumseagreen','orange'))
plt.legend()
plt.xlabel('Dice Score')
plt.ylabel('# Number of slices')
plt.title('Histogram of Dice Scores for dia model')
plt.grid(True, color = "grey", linewidth = "0.5", linestyle = "-")

#%%

data_SD_dia  = dice_sys
data_CE_dia   = dice_dia
#%%
data_SD_sys  = dice_sys
data_CE_sys   = dice_dia
#%%

#fig, ax = plt.subplots()
plt.figure(dpi=200, figsize=(14,7))
plt.subplot(1,2,1)
bp1 = plt.boxplot(data_CE_dia, positions=[1,3,5], widths=0.35, 
                 patch_artist=True, boxprops=dict(facecolor="C0"),medianprops=dict(color='red'))
bp2 = plt.boxplot(data_SD_dia, positions=[2,4,6], widths=0.35, 
                 patch_artist=True, boxprops=dict(facecolor="C2"),medianprops=dict(color='red'))

plt.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Cross-Entropy', 'Soft-Dice'], fontsize=12, loc='lower right')
plt.xticks(np.arange(1,7),['RV','RV','MYO','MYO','LV','LV'] , fontsize=12)
plt.ylabel('Dice Score', fontsize=12)
plt.title('Boxplot of Dice scores for diastolic data', fontsize=15)
plt.xlim(0.5,7.2)
plt.ylim(-0.15,1.1)

plt.subplot(1,2,2)
bp1 = plt.boxplot(data_CE_sys, positions=[1,3,5], widths=0.35, 
                 patch_artist=True, boxprops=dict(facecolor="C0"),medianprops=dict(color='red'))
bp2 = plt.boxplot(data_SD_sys, positions=[2,4,6], widths=0.35, 
                 patch_artist=True, boxprops=dict(facecolor="C2"),medianprops=dict(color='red'))

plt.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Cross-Entropy', 'Soft-Dice'], fontsize=12, loc='lower right')
plt.xticks(np.arange(1,7),['RV','RV','MYO','MYO','LV','LV'],  fontsize=12 )
plt.ylabel('Dice Score', fontsize=12)
plt.title('Boxplot of Dice scores for systolic data', fontsize=15)
plt.xlim(0.5,7.2)
plt.ylim(-0.15,1.1)
plt.show()

#%%%

myo = soft_sys[:,31,:,:,:]
myo_am = np.argmax(myo, axis=1)
seg = torch.nn.functional.one_hot(torch.as_tensor(myo_am), num_classes=4).detach().cpu().numpy()
seg = seg[:,:,:,2]


seg = np.sum(seg, axis=0)

plt.figure(dpi=200)
plt.imshow(seg)
plt.colorbar()



#%%




























