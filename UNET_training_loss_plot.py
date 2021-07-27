# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 09:05:26 2021

@author: katrine
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 11:51:36 2021

@author: michalablicher
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
#!pip install torch-summary
#!pip install opencv-python

#%% Import results from training (Loss + Accuracy)
#PATH_dice = 'C:/Users/katrine/Desktop/Optuna/Final CV models/Trained_Unet_dice_lclv_dia_150e_train_results.pt'
#PATH_CE   = 'C:/Users/katrine/Desktop/Optuna/Final CV models/Trained_Unet_dice_lclv_dia_150e_opt_train_results.pt'
PATH_dice = '/Users/michalablicher/Desktop/Trained_Unet_dice_lclv_dia_150e_train_results.pt'
PATH_CE   = '/Users/michalablicher/Desktop/Trained_Unet_dice_lclv_dia_150e_opt_train_results.pt'

res_dice = torch.load(PATH_dice, map_location=torch.device('cpu'))
res_CE = torch.load(PATH_CE, map_location=torch.device('cpu'))

#%% Loss and accuracy
out_mean = res_dice[0] # import mean from model
out_one  = res_dice[1] 

train_loss = out_mean[0]
eval_loss  = out_mean[1]

train_loss_0 = out_one[0][0]
eval_loss_0  = out_one[1][0]
train_loss_1 = out_one[0][1]
eval_loss_1  = out_one[1][1]
train_loss_2 = out_one[0][2]
eval_loss_2  = out_one[1][2]
train_loss_3 = out_one[0][3]
eval_loss_3  = out_one[1][3]
train_loss_4 = out_one[0][4]
eval_loss_4  = out_one[1][4]
train_loss_5 = out_one[0][5]
eval_loss_5  = out_one[1][5]

train_acc = out_mean[2]
eval_acc  = out_mean[3]

train_acc_1 = out_one[2][0]
eval_acc_1  = out_one[3][0]
train_acc_2 = out_one[2][1]
eval_acc_2  = out_one[3][1]
train_acc_3 = out_one[2][2]
eval_acc_3  = out_one[3][2]
train_acc_4 = out_one[2][3]
eval_acc_4  = out_one[3][3]
train_acc_5 = out_one[2][4]
eval_acc_5  = out_one[3][4]
train_acc_6 = out_one[2][5]
eval_acc_6  = out_one[3][5]

train_inc = (out_mean[4])/5
eval_inc  = (out_mean[5])

#%% CE
out_mean_CE = res_CE[0] # import mean from model
out_one_CE  = res_CE[1] 

train_loss_CE = out_mean_CE[0]
eval_loss_CE  = out_mean_CE[1]

train_loss_0_CE = out_one_CE[0][0]
eval_loss_0_CE  = out_one_CE[1][0]
train_loss_1_CE = out_one_CE[0][1]
eval_loss_1_CE  = out_one_CE[1][1]
train_loss_2_CE = out_one_CE[0][2]
eval_loss_2_CE  = out_one_CE[1][2]
train_loss_3_CE = out_one_CE[0][3]
eval_loss_3_CE  = out_one_CE[1][3]
train_loss_4_CE = out_one_CE[0][4]
eval_loss_4_CE  = out_one_CE[1][4]
train_loss_5_CE = out_one_CE[0][5]
eval_loss_5_CE  = out_one_CE[1][5]

train_acc_CE = out_mean_CE[2]
eval_acc_CE  = out_mean_CE[3]

train_acc_1_CE = out_one_CE[2][0]
eval_acc_1_CE  = out_one_CE[3][0]
train_acc_2_CE = out_one_CE[2][1]
eval_acc_2_CE  = out_one_CE[3][1]
train_acc_3_CE = out_one_CE[2][2]
eval_acc_3_CE  = out_one_CE[3][2]
train_acc_4_CE = out_one_CE[2][3]
eval_acc_4_CE  = out_one_CE[3][3]
train_acc_5_CE = out_one_CE[2][4]
eval_acc_5_CE  = out_one_CE[3][4]
train_acc_6_CE = out_one_CE[2][5]
eval_acc_6_CE  = out_one_CE[3][5]

train_inc_CE = (out_mean_CE[4])/5
eval_inc_CE  = (out_mean_CE[5])
#%%


t1 = np.arange(0,50) #np.arange(len(train_acc_1))
t2 = np.arange(len(train_acc_1_CE))

plt.figure(figsize=(12, 12),dpi=400)
plt.plot(t1, train_acc_1[0:50], 'b', label = 'Training Loss')
plt.plot(t1 , train_acc_2[0:50], 'g', label = 'Training Loss')
plt.plot(t1 , train_acc_3[0:50], 'r', label = 'Training Loss')
plt.plot(t1 , train_acc_4[0:50], 'y', label = 'Training Loss')
plt.plot(t1 , train_acc_5[0:50], 'm', label = 'Training Loss')
plt.plot(t1 , train_acc_6[0:50], 'c', label = 'Training Loss')

plt.plot(t1, eval_acc_1[0:50], 'b' ,linestyle = 'dashed', label = 'Validation Loss')
plt.plot(t1 , eval_acc_2[0:50], 'g',linestyle = 'dashed', label = 'Validation Loss')
plt.plot(t1 , eval_acc_3[0:50], 'r',linestyle = 'dashed', label = 'Validation Loss')
plt.plot(t1 , eval_acc_4[0:50], 'y',linestyle = 'dashed', label = 'Validation Loss')
plt.plot(t1 , eval_acc_5[0:50], 'm',linestyle = 'dashed', label = 'Validation Loss')
plt.plot(t1 , eval_acc_6[0:50], 'c',linestyle = 'dashed', label = 'Validation Loss')


#%% Plot function
epochs_train = np.arange(len(train_loss))
epochs_eval  = np.arange(len(eval_loss))

t1 = epochs_train #np.arange(0,50) 
plt.figure(figsize=(15, 15),dpi=400)

plt.subplot(2,2,1)
plt.plot(t1, train_loss_0, 'b', label = 'Training loss fold 0')
plt.plot(t1 , train_loss_1, 'g', label = 'Training loss fold 1')
plt.plot(t1 , train_loss_2, 'r', label = 'Training loss fold 2')
plt.plot(t1 , train_loss_3, 'y', label = 'Training loss fold 3')
plt.plot(t1 , train_loss_4, 'm', label = 'Training loss fold 4')
plt.plot(t1 , train_loss_5, 'c', label = 'Training loss fold 5')

plt.plot(t1, eval_loss_0, 'b' ,linestyle = 'dashed', label = 'Validation loss fold 0')
plt.plot(t1 , eval_loss_1, 'g',linestyle = 'dashed', label = 'Validation loss fold 1')
plt.plot(t1 , eval_loss_2, 'r',linestyle = 'dashed', label = 'Validation loss fold 2')
plt.plot(t1 , eval_loss_3, 'y',linestyle = 'dashed', label = 'Validation loss fold 3')
plt.plot(t1 , eval_loss_4, 'm',linestyle = 'dashed', label = 'Validation loss fold 4')
plt.plot(t1 , eval_loss_5, 'c',linestyle = 'dashed', label = 'Validation loss fold 5')

#plt.scatter(np.argmin(eval_loss),np.min(eval_loss),marker="v",s=10)

plt.grid(color='k', linestyle='-', linewidth=0.2)
plt.xticks(np.arange(0, len(t1) + 2, step = 50), fontsize =14)
plt.yticks(fontsize =14)
plt.xlabel('Epochs', fontsize = 16)
plt.ylabel('$Loss_D + Loss_N + Loss_C$',  fontsize = 16)
plt.legend(loc="upper right", fontsize = 13)
plt.title('Loss curve (AE)', fontsize =28)

plt.subplot(2,2,2)
plt.plot(t1, train_acc_1, 'b', label = 'Training accuracy fold 0')
plt.plot(t1 , train_acc_2, 'g', label = 'Training accuracy fold 1')
plt.plot(t1 , train_acc_3, 'r', label = 'Training accuracy fold 2')
plt.plot(t1 , train_acc_4, 'y', label = 'Training accuracy fold 3')
plt.plot(t1 , train_acc_5, 'm', label = 'Training accuracy fold 4')
plt.plot(t1 , train_acc_6, 'c', label = 'Training accuracy fold 5')

plt.plot(t1, eval_acc_1, 'b' ,linestyle = 'dashed', label = 'Validation accuracy fold 0')
plt.plot(t1 , eval_acc_2, 'g',linestyle = 'dashed', label = 'Validation accuracy fold 1')
plt.plot(t1 , eval_acc_3, 'r',linestyle = 'dashed', label = 'Validation accuracy fold 2')
plt.plot(t1 , eval_acc_4, 'y',linestyle = 'dashed', label = 'Validation accuracy fold 3')
plt.plot(t1 , eval_acc_5, 'm',linestyle = 'dashed', label = 'Validation accuracy fold 4')
plt.plot(t1 , eval_acc_6, 'c',linestyle = 'dashed', label = 'Validation accuracy fold 5')

plt.grid(color='k', linestyle='-', linewidth=0.2)
plt.xticks(np.arange(0, len(t1) + 2, step = 50), fontsize =14)
plt.yticks(fontsize =14)
plt.xlabel('Epochs', fontsize = 16)
plt.ylabel('Accuracy %',  fontsize = 16)
plt.legend(loc="lower right", fontsize = 13)
plt.title("Accuracy (AE)", fontsize =28)



plt.subplot(2,2,3)
plt.plot(t2, train_loss_0_CE, 'b', label = 'Training loss fold 0')
plt.plot(t2 , train_loss_1_CE, 'g', label = 'Training loss fold 1')
plt.plot(t2 , train_loss_2_CE, 'r', label = 'Training loss fold 2')
plt.plot(t2 , train_loss_3_CE, 'y', label = 'Training loss fold 3')
plt.plot(t2 , train_loss_4_CE, 'm', label = 'Training loss fold 4')
plt.plot(t2 , train_loss_5_CE, 'c', label = 'Training loss fold 5')

plt.plot(t2, eval_loss_0_CE, 'b' ,linestyle = 'dashed', label = 'Validation loss fold 0')
plt.plot(t2 , eval_loss_1_CE, 'g',linestyle = 'dashed', label = 'Validation loss fold 1')
plt.plot(t2 , eval_loss_2_CE, 'r',linestyle = 'dashed', label = 'Validation loss fold 2')
plt.plot(t2 , eval_loss_3_CE, 'y',linestyle = 'dashed', label = 'Validation loss fold 3')
plt.plot(t2 , eval_loss_4_CE, 'm',linestyle = 'dashed', label = 'Validation loss fold 4')
plt.plot(t2 , eval_loss_5_CE, 'c',linestyle = 'dashed', label = 'Validation loss fold 5')

#plt.scatter(np.argmin(eval_loss),np.min(eval_loss),marker="v",s=10)

plt.grid(color='k', linestyle='-', linewidth=0.2)
plt.xticks(np.arange(0, len(t1) + 2, step = 50), fontsize =14)
plt.yticks(fontsize =14)
plt.xlabel('Epochs', fontsize = 16)
plt.ylabel(r'$Loss_D + \alpha Loss_N + \beta Loss_C$',  fontsize = 16)
plt.legend(loc="upper right", fontsize = 13)
plt.title('Loss curve (AE opt)', fontsize =28)

plt.subplot(2,2,4)
plt.plot(t2, train_acc_1_CE, 'b', label = 'Training accuracy fold 0')
plt.plot(t2 , train_acc_2_CE, 'g', label = 'Training accuracy fold 1')
plt.plot(t2 , train_acc_3_CE, 'r', label = 'Training accuracy fold 2')
plt.plot(t2 , train_acc_4_CE, 'y', label = 'Training accuracy fold 3')
plt.plot(t2 , train_acc_5_CE, 'm', label = 'Training accuracy fold 4')
plt.plot(t2 , train_acc_6_CE, 'c', label = 'Training accuracy fold 5')

plt.plot(t2, eval_acc_1_CE, 'b' ,linestyle = 'dashed', label = 'Validation accuracy fold 0')
plt.plot(t2 , eval_acc_2_CE, 'g',linestyle = 'dashed', label = 'Validation accuracy fold 1')
plt.plot(t2 , eval_acc_3_CE, 'r',linestyle = 'dashed', label = 'Validation accuracy fold 2')
plt.plot(t2 , eval_acc_4_CE, 'y',linestyle = 'dashed', label = 'Validation accuracy fold 3')
plt.plot(t2 , eval_acc_5_CE, 'm',linestyle = 'dashed', label = 'Validation accuracy fold 4')
plt.plot(t2 , eval_acc_6_CE, 'c',linestyle = 'dashed', label = 'Validation accuracy fold 5')

plt.grid(color='k', linestyle='-', linewidth=0.2)
plt.xticks(np.arange(0, len(t1) + 2, step = 50), fontsize =14)
plt.yticks(fontsize =14)
plt.xlabel('Epochs', fontsize = 16)
plt.ylabel('Accuracy %',  fontsize = 16)
plt.legend(loc="lower right", fontsize = 13)
plt.title("Accuracy (AE opt)", fontsize =28)

#%%

#%% Plot function
t1 = np.arange(0,50) #np.arange(len(train_loss))
epochs_eval  = np.arange(0,50) #np.arange(len(eval_loss))

plt.figure(figsize=(13, 5),dpi=400)

plt.subplot(1,2,1)
plt.plot(t1, train_loss_0[0:50], 'b', label = 'Training loss fold 0')
plt.plot(t1 , train_loss_1[0:50], 'g', label = 'Training loss fold 1')
plt.plot(t1 , train_loss_2[0:50], 'r', label = 'Training loss fold 2')
plt.plot(t1 , train_loss_3[0:50], 'y', label = 'Training loss fold 3')
plt.plot(t1 , train_loss_4[0:50], 'm', label = 'Training loss fold 4')
plt.plot(t1 , train_loss_5[0:50], 'c', label = 'Training loss fold 5')

plt.subplot(1,2,2)
plt.plot(t1, eval_loss_0[0:50], 'b' ,linestyle = 'dashed', label = 'Validation loss fold 0')
plt.plot(t1 , eval_loss_1[0:50], 'g',linestyle = 'dashed', label = 'Validation loss fold 1')
plt.plot(t1 , eval_loss_2[0:50], 'r',linestyle = 'dashed', label = 'Validation loss fold 2')
plt.plot(t1 , eval_loss_3[0:50], 'y',linestyle = 'dashed', label = 'Validation loss fold 3')
plt.plot(t1 , eval_loss_4[0:50], 'm',linestyle = 'dashed', label = 'Validation loss fold 4')
plt.plot(t1 , eval_loss_5[0:50], 'c',linestyle = 'dashed', label = 'Validation loss fold 5')

#plt.scatter(np.argmin(eval_loss),np.min(eval_loss),marker="v",s=10)

plt.grid(color='k', linestyle='-', linewidth=0.2)
plt.xticks(np.arange(0, len(t1) + 2, step = 50), fontsize =14)
plt.yticks(fontsize =14)
plt.xlabel('Epochs', fontsize = 16)
plt.ylabel('$Loss_D + Loss_N + Loss_C$',  fontsize = 16)
plt.legend(loc="upper right", fontsize = 13)
plt.title('Loss curve (AE)', fontsize =28)
#%%
plt.figure(figsize=(13, 5),dpi=400)

plt.subplot(1,2,1)
plt.plot(t1, train_acc_1, 'b', label = 'Training accuracy fold 0')
plt.plot(t1 , train_acc_2, 'g', label = 'Training accuracy fold 1')
plt.plot(t1 , train_acc_3, 'r', label = 'Training accuracy fold 2')
plt.plot(t1 , train_acc_4, 'y', label = 'Training accuracy fold 3')
plt.plot(t1 , train_acc_5, 'm', label = 'Training accuracy fold 4')
plt.plot(t1 , train_acc_6, 'c', label = 'Training accuracy fold 5')

plt.subplot(1,2,2)
plt.plot(t1, eval_acc_1, 'b' ,linestyle = 'dashed', label = 'Validation accuracy fold 0')
plt.plot(t1 , eval_acc_2, 'g',linestyle = 'dashed', label = 'Validation accuracy fold 1')
plt.plot(t1 , eval_acc_3, 'r',linestyle = 'dashed', label = 'Validation accuracy fold 2')
plt.plot(t1 , eval_acc_4, 'y',linestyle = 'dashed', label = 'Validation accuracy fold 3')
plt.plot(t1 , eval_acc_5, 'm',linestyle = 'dashed', label = 'Validation accuracy fold 4')
plt.plot(t1 , eval_acc_6, 'c',linestyle = 'dashed', label = 'Validation accuracy fold 5')

plt.grid(color='k', linestyle='-', linewidth=0.2)
plt.xticks(np.arange(0, len(t1) + 2, step = 50), fontsize =14)
plt.yticks(fontsize =14)
plt.xlabel('Epochs', fontsize = 16)
plt.ylabel('Accuracy %',  fontsize = 16)
plt.legend(loc="lower right", fontsize = 13)
plt.title("Accuracy (AE)", fontsize =28)

#%%

plt.subplot(2,2,3)
plt.plot(t2, train_loss_0_CE, 'b', label = 'Training loss fold 0')
plt.plot(t2 , train_loss_1_CE, 'g', label = 'Training loss fold 1')
plt.plot(t2 , train_loss_2_CE, 'r', label = 'Training loss fold 2')
plt.plot(t2 , train_loss_3_CE, 'y', label = 'Training loss fold 3')
plt.plot(t2 , train_loss_4_CE, 'm', label = 'Training loss fold 4')
plt.plot(t2 , train_loss_5_CE, 'c', label = 'Training loss fold 5')

plt.plot(t2, eval_loss_0_CE, 'b' ,linestyle = 'dashed', label = 'Validation loss fold 0')
plt.plot(t2 , eval_loss_1_CE, 'g',linestyle = 'dashed', label = 'Validation loss fold 1')
plt.plot(t2 , eval_loss_2_CE, 'r',linestyle = 'dashed', label = 'Validation loss fold 2')
plt.plot(t2 , eval_loss_3_CE, 'y',linestyle = 'dashed', label = 'Validation loss fold 3')
plt.plot(t2 , eval_loss_4_CE, 'm',linestyle = 'dashed', label = 'Validation loss fold 4')
plt.plot(t2 , eval_loss_5_CE, 'c',linestyle = 'dashed', label = 'Validation loss fold 5')

#plt.scatter(np.argmin(eval_loss),np.min(eval_loss),marker="v",s=10)

plt.grid(color='k', linestyle='-', linewidth=0.2)
plt.xticks(np.arange(0, len(t1) + 2, step = 50), fontsize =14)
plt.yticks(fontsize =14)
plt.xlabel('Epochs', fontsize = 16)
plt.ylabel(r'$Loss_D + \alpha Loss_N + \beta Loss_C$',  fontsize = 16)
plt.legend(loc="upper right", fontsize = 13)
plt.title('Loss curve (AE opt)', fontsize =28)

plt.subplot(2,2,4)
plt.plot(t2, train_acc_1_CE, 'b', label = 'Training accuracy fold 0')
plt.plot(t2 , train_acc_2_CE, 'g', label = 'Training accuracy fold 1')
plt.plot(t2 , train_acc_3_CE, 'r', label = 'Training accuracy fold 2')
plt.plot(t2 , train_acc_4_CE, 'y', label = 'Training accuracy fold 3')
plt.plot(t2 , train_acc_5_CE, 'm', label = 'Training accuracy fold 4')
plt.plot(t2 , train_acc_6_CE, 'c', label = 'Training accuracy fold 5')

plt.plot(t2, eval_acc_1_CE, 'b' ,linestyle = 'dashed', label = 'Validation accuracy fold 0')
plt.plot(t2 , eval_acc_2_CE, 'g',linestyle = 'dashed', label = 'Validation accuracy fold 1')
plt.plot(t2 , eval_acc_3_CE, 'r',linestyle = 'dashed', label = 'Validation accuracy fold 2')
plt.plot(t2 , eval_acc_4_CE, 'y',linestyle = 'dashed', label = 'Validation accuracy fold 3')
plt.plot(t2 , eval_acc_5_CE, 'm',linestyle = 'dashed', label = 'Validation accuracy fold 4')
plt.plot(t2 , eval_acc_6_CE, 'c',linestyle = 'dashed', label = 'Validation accuracy fold 5')

plt.grid(color='k', linestyle='-', linewidth=0.2)
plt.xticks(np.arange(0, len(t1) + 2, step = 50), fontsize =14)
plt.yticks(fontsize =14)
plt.xlabel('Epochs', fontsize = 16)
plt.ylabel('Accuracy %',  fontsize = 16)
plt.legend(loc="lower right", fontsize = 13)
plt.title("Accuracy (AE opt)", fontsize =28)






#%%
plt.figure(dpi=200)
plt.semilogy(epochs_train + 1 , train_inc, 'b', label = 'Training incorrect')
plt.semilogy(epochs_eval  + 1 , eval_inc,  'r' ,label = 'Validation incorrect')
plt.grid(color='k', linestyle='-', linewidth=0.2)
plt.xticks(np.arange(0, len(epochs_train) + 2, step = 25), fontsize =14)
plt.yticks(fontsize =14)
plt.xlabel('Epochs', fontsize = 16)
plt.ylabel('Log #incorrect seg',  fontsize = 16)
plt.legend(loc="upper right", fontsize = 13)
plt.title("Number of Incorrect", fontsize =28)



#%% Plot function
epochs_train = np.arange(len(train_loss))
epochs_eval  = np.arange(len(eval_loss))

epochs_train_CE = np.arange(len(train_loss_CE))
epochs_eval_CE  = np.arange(len(eval_loss_CE))

plt.figure(figsize=(30, 20),dpi=400)
#plt.rcParams.update({'font.size': 26})
plt.subplot(2,3,1)
plt.plot(epochs_train + 1 , train_loss, 'b', label = 'Training Loss    (AE)', linewidth=3)
#plt.plot(epochs_eval  + 1 , eval_loss, 'r', label = 'Validation Loss (AE)')
plt.plot(epochs_train_CE + 1 , train_loss_CE, 'g', label = 'Training Loss    (AE opt)', linewidth=3)
#plt.plot(epochs_eval_CE  + 1 , eval_loss_CE, 'k', label = 'Validation Loss (AE opt)')

plt.grid(color='k', linestyle='-', linewidth=0.2)
plt.xticks(np.arange(0, len(epochs_train) + 2, step = 50), fontsize =18)
plt.yticks(np.arange(0, 2.5, step=0.5), fontsize =18)
plt.xlabel('Epochs', fontsize = 25)
plt.ylabel(r'$Loss_D + \alpha Loss_N + \beta Loss_C$',  fontsize = 25)
plt.legend(loc="upper right", fontsize = 25)
plt.title('Loss function for averaged models', fontsize =28)

plt.subplot(2,3,2)
plt.plot(epochs_train + 1 , train_acc, 'b', label = 'Training accuracy    (AE)', linewidth=3)
#plt.plot(epochs_eval  + 1 , eval_acc,  'r',label = 'Validation accuracy (AE)')
plt.plot(epochs_train_CE + 1 , train_acc_CE, 'g', label = 'Training accuracy    (AE opt)', linewidth=3)
#plt.plot(epochs_eval_CE  + 1 , eval_acc_CE,  'k',label = 'Validation accuracy (AE opt)')

plt.grid(color='k', linestyle='-', linewidth=0.2)
plt.xticks(np.arange(0, len(epochs_train) + 2, step = 50), fontsize =18)
plt.yticks(fontsize =18)
plt.xlabel('Epochs', fontsize = 25)
plt.ylabel('Accuracy %',  fontsize = 25)
plt.legend(loc="lower right", fontsize = 25)
plt.title("Accuracy for averaged models", fontsize =28)

plt.subplot(2,3,3)
plt.semilogy(epochs_train + 1 , train_inc, 'b', label = 'Training incorrect    (AE)', linewidth=3)
#plt.semilogy(epochs_eval  + 1 , eval_inc,  'r' ,label = 'Validation incorrect (AE)')
plt.semilogy(epochs_train_CE + 1 , train_inc_CE, 'g', label = 'Training incorrect    (AE opt)', linewidth=3)
#plt.semilogy(epochs_eval_CE  + 1 , eval_inc_CE,  'k' ,label = 'Validation incorrect (AE opt)')

plt.grid(color='k', linestyle='-', linewidth=0.2)
plt.xticks(np.arange(0, len(epochs_train) + 2, step = 50), fontsize =18)
plt.yticks(fontsize = 18)
plt.xlabel('Epochs', fontsize = 25)
plt.ylabel('Log #incorrect seg',  fontsize = 25)
plt.legend(loc="upper right", fontsize = 25)
plt.title("Number of incorrect for averaged models", fontsize =28)

plt.subplot(2,3,4)
#plt.plot(epochs_train + 1 , train_loss, 'b', label = 'Training Loss    (AE)')
plt.plot(epochs_eval  + 1 , eval_loss, 'r', label = 'Validation Loss (AE)', linewidth=3)
#plt.plot(epochs_train_CE + 1 , train_loss_CE, 'g', label = 'Training Loss    (AE opt)')
plt.plot(epochs_eval_CE  + 1 , eval_loss_CE, 'k', label = 'Validation Loss (AE opt)', linewidth=3)

plt.grid(color='k', linestyle='-', linewidth=0.2)
plt.xticks(np.arange(0, len(epochs_train) + 2, step = 50), fontsize =18)
plt.yticks(np.arange(0, 2.51, step=0.5), fontsize =18)
plt.ylim([-0.05,2.4])
plt.xlabel('Epochs', fontsize = 25)
plt.ylabel(r'$Loss_D + \alpha Loss_N + \beta Loss_C$',  fontsize = 25)
plt.legend(loc="upper right", fontsize = 25)
plt.title('Loss function for averaged models', fontsize =28)

plt.subplot(2,3,5)
#plt.plot(epochs_train + 1 , train_acc, 'b', label = 'Training accuracy    (AE)')
plt.plot(epochs_eval  + 1 , eval_acc,  'r',label = 'Validation accuracy (AE)', linewidth=3)
#plt.plot(epochs_train_CE + 1 , train_acc_CE, 'g', label = 'Training accuracy    (AE opt)')
plt.plot(epochs_eval_CE  + 1 , eval_acc_CE,  'k',label = 'Validation accuracy (AE opt)', linewidth=3)

plt.grid(color='k', linestyle='-', linewidth=0.2)
plt.xticks(np.arange(0, len(epochs_train) + 2, step = 50), fontsize =18)
plt.yticks(np.arange(60, 100 + 1, step=5), fontsize =18)
plt.xlabel('Epochs', fontsize = 25)
plt.ylabel('Accuracy %',  fontsize = 25)
plt.legend(loc="lower right", fontsize = 25)
plt.title("Accuracy for averaged models", fontsize =28)

plt.subplot(2,3,6)
#plt.semilogy(epochs_train + 1 , train_inc, 'b', label = 'Training incorrect    (AE)')
plt.semilogy(epochs_eval  + 1 , eval_inc,  'r' ,label = 'Validation incorrect (AE)', linewidth=3)
#plt.semilogy(epochs_train_CE + 1 , train_inc_CE, 'g', label = 'Training incorrect    (AE opt)')
plt.semilogy(epochs_eval_CE  + 1 , eval_inc_CE,  'k' ,label = 'Validation incorrect (AE opt)', linewidth=3)

plt.grid(color='k', linestyle='-', linewidth=0.2)
plt.xticks(np.arange(0, len(epochs_train) + 2, step = 50), fontsize =18)
plt.yticks(fontsize = 18)
plt.xlabel('Epochs', fontsize = 25)
plt.ylabel('Log #incorrect seg',  fontsize = 25)
plt.legend(loc="upper right", fontsize = 25)
plt.title("Number of incorrect for averaged models", fontsize =28)

##################################################################################################################
##################################################################################################################
#%% Import results from training (Loss + Accuracy)
PATH_dice = 'C:/Users/katrine/Desktop/Optuna/Final CV models/Out_softmax_fold_avg_150dia_dice_lclv.pt'
PATH_CE   = 'C:/Users/katrine/Desktop/Optuna/Final CV models/Out_softmax_fold_avg_150dia_dice_lclv_opt.pt'
#PATH_res_ed = '/Users/michalablicher/Desktop/Trained_Unet_dicew_2lv_dia_200e_train_results.pt'
res_dice = torch.load(PATH_dice, map_location=torch.device('cpu'))
res_CE = torch.load(PATH_CE, map_location=torch.device('cpu'))
##################################################################################################################
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

phase = 'Systole'
data_im_es_DCM,  data_gt_es_DCM  = load_data_sub(user,phase,'DCM')
data_im_es_HCM,  data_gt_es_HCM  = load_data_sub(user,phase,'HCM')
data_im_es_MINF, data_gt_es_MINF = load_data_sub(user,phase,'MINF')
data_im_es_NOR,  data_gt_es_NOR  = load_data_sub(user,phase,'NOR')
data_im_es_RV,   data_gt_es_RV   = load_data_sub(user,phase,'RV')
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
#%% Plot softmax probabilities for all CV models
#Plot softmax probabilities for a single slice
test_slice = 145
alpha = 0.4

model = 'CE'
#model = 'SD'

if model == 'CE':      # lclv
    out_soft = res_CE
if model == 'SD':      # SD
    out_soft = res_dice

fig = plt.figure()

class_title = ['Background','Right Ventricle','Myocardium','Left Ventricle']
ref_dia = torch.nn.functional.one_hot(Tensor(gt_test_ed_sub).to(torch.int64), num_classes=4)
plt.figure(dpi=200, figsize=(18,32))

w = 0.1

for fold_model in range (0,6):
    out_img_ed = np.squeeze(out_soft[fold_model,:,:,:,:])
    #seg_met_dia = np.argmax(out_soft[fold_model,:,:,:], axis=1)
    #seg_dia = torch.nn.functional.one_hot(torch.as_tensor(seg_met_dia), num_classes=4)
    seg_dia = Tensor(out_img_ed).permute(0,2,3,1).detach().numpy()
    
    #Reference annotation
    #plt.suptitle('Softmax probabilities for each model at test slice %i (SD)' %test_slice, fontsize=35, y=0.92)
    plt.suptitle('Softmax probabilities for each model at test slice %i (Lclv)' %test_slice, fontsize=35, y=0.92)
    plt.subplot(7, 4, 1)
    plt.subplots_adjust(hspace = 0.05, wspace = w)
    plt.imshow(ref_dia[test_slice,:,:,0])
    plt.imshow(im_test_ed_sub[test_slice,0,:,:],alpha=alpha)
    plt.ylabel('Reference', fontsize=28)
    plt.title('Background', fontsize=28)
    
    plt.subplot(7, 4, 2)
    plt.subplots_adjust(hspace = 0.05, wspace = w)
    plt.imshow(ref_dia[test_slice,:,:,1])
    plt.imshow(im_test_ed_sub[test_slice,0,:,:],alpha=alpha)
    plt.title('Right ventricle', fontsize=28)
    
    plt.subplot(7, 4, 3)
    plt.subplots_adjust(hspace = 0.05, wspace = w)
    plt.imshow(ref_dia[test_slice,:,:,2])
    plt.imshow(im_test_ed_sub[test_slice,0,:,:],alpha=alpha)
    plt.title('Myocardium', fontsize=28)
    
    plt.subplot(7, 4, 4)
    plt.subplots_adjust(hspace = 0.05, wspace = w)
    plt.imshow(ref_dia[test_slice,:,:,3])
    plt.imshow(im_test_ed_sub[test_slice,0,:,:],alpha=alpha)
    plt.title('Left ventricle', fontsize=28)
    
    
    #CV model segmentations
    plt.subplot(7, 4, 1+4*(fold_model+1))
    plt.subplots_adjust(hspace = 0.05, wspace = w)
    plt.imshow(seg_dia[test_slice,:,:,0])
    plt.imshow(im_test_ed_sub[test_slice,0,:,:],alpha=alpha)
    plt.ylabel('CV fold {}'.format(fold_model), fontsize=28)
    
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

plt.show()  
#%% Averaged model
#test_slice = 27
alpha = 0.4

model = 'CE'

if model == 'CE':
    out_soft = res_CE
if model == 'SD':
    out_soft = res_dice

out_soft_mean   = out_soft.mean(axis=0)

plt.figure(dpi=300, figsize=(1.5*5,3.5*5))
plt.suptitle('Softmax probabilies for averaged model at test slice {} (LcLv)'.format(test_slice), y=0.92, fontsize=18)
for i in range(0,4):
    plt.subplot(4,2,i+1)
    plt.imshow(out_soft_mean[test_slice,i,:,:])
    plt.title(class_title[i], fontsize=15)

plt.show()

#%% Averaged model
alpha = 0.4

model = 'SD'

if model == 'CE':
    out_soft = res_CE
if model == 'SD':
    out_soft = res_dice

out_soft_mean   = out_soft.mean(axis=0)

plt.figure(dpi=300, figsize=(1.5*5,3.5*5))
plt.suptitle('Softmax probabilies for averaged model at test slice {} (SD)'.format(test_slice), y=0.92, fontsize=18)
for i in range(0,4):
    plt.subplot(4,2,i+1)
    plt.imshow(out_soft_mean[test_slice,i,:,:])
    plt.title(class_title[i], fontsize=15)

plt.show()

#%%
s= 8
test_slice = 66
plt.figure(dpi=400, figsize=(s,s))
plt.imshow(out_soft_mean[test_slice,2,:,:])
plt.colorbar(orientation='horizontal', ticks=[0,0.2,0.4,0.6,0.8,1.0])
#%% Argmax model
out_seg_mean_am = np.argmax(out_soft_mean, axis=1)
out_seg_mean    = torch.nn.functional.one_hot(torch.as_tensor(out_seg_mean_am), num_classes=4).detach().cpu().numpy()

plt.figure(dpi=300, figsize=(10,7))
plt.suptitle('Segmentations for averaged models at test slice 31', y=0.98, fontsize=20)

plt.subplot(2,3,1)
plt.imshow(im_test_ed_sub[test_slice,0,:,:])
plt.title('Original cMRI', fontsize=15)
plt.ylabel('Diastolic', fontsize=15)

plt.subplot(2,3,4)
plt.imshow(im_test_es_sub[test_slice,0,:,:])
plt.title('Original cMRI', fontsize=15)
plt.ylabel('Opt', fontsize=15)

plt.subplot(2,3,2)
out_soft = res_dice
out_soft_mean   = out_soft.mean(axis=0)
out_seg_mean_am = np.argmax(out_soft_mean, axis=1)
plt.imshow(out_seg_mean_am[test_slice,:,:])
plt.imshow(im_test_ed_sub[test_slice,0,:,:],alpha=alpha)
plt.title('Segmentation', fontsize=15)

plt.subplot(2,3,5)
out_soft = res_CE
out_soft_mean   = out_soft.mean(axis=0)
out_seg_mean_am = np.argmax(out_soft_mean, axis=1)
plt.imshow(out_seg_mean_am[test_slice,:,:])
plt.imshow(im_test_es_sub[test_slice,0,:,:],alpha=alpha)
plt.title('Segmentation', fontsize=15)

plt.subplot(2,3,3)
plt.imshow(gt_test_ed_sub[test_slice,:,:])
plt.imshow(im_test_ed_sub[test_slice,0,:,:],alpha=alpha)
plt.title('Reference', fontsize=15)

plt.subplot(2,3,6)
plt.imshow(gt_test_es_sub[test_slice,:,:])
plt.imshow(im_test_es_sub[test_slice,0,:,:],alpha=alpha)
plt.title('Reference', fontsize=15)

#%% Argmax model
out_seg_mean_am = np.argmax(out_soft_mean, axis=1)
out_seg_mean    = torch.nn.functional.one_hot(torch.as_tensor(out_seg_mean_am), num_classes=4).detach().cpu().numpy()

plt.figure(dpi=300, figsize=(12,7))
plt.suptitle('Segmentations for averaged models at test slice 31', y=0.75, fontsize=20)

plt.subplot(1,4,1)
plt.imshow(im_test_ed_sub[test_slice,0,:,:])
plt.title('Original cMRI', fontsize=15)

plt.subplot(1,4,3)
out_soft = res_dice
out_soft_mean   = out_soft.mean(axis=0)
out_seg_mean_am = np.argmax(out_soft_mean, axis=1)
plt.imshow(out_seg_mean_am[test_slice,:,:])
plt.imshow(im_test_ed_sub[test_slice,0,:,:],alpha=alpha)
plt.title('Soft-Dice', fontsize=15)

plt.subplot(1,4,2)
out_soft = res_CE
out_soft_mean   = out_soft.mean(axis=0)
out_seg_mean_am = np.argmax(out_soft_mean, axis=1)
plt.imshow(out_seg_mean_am[test_slice,:,:])
plt.imshow(im_test_es_sub[test_slice,0,:,:],alpha=alpha)
plt.title('Cross-Entropy', fontsize=15)

plt.subplot(1,4,4)
plt.imshow(gt_test_ed_sub[test_slice,:,:])
plt.imshow(im_test_ed_sub[test_slice,0,:,:],alpha=alpha)
plt.title('Reference', fontsize=15)


#%%%%%%%%%%%%%%%%%%%%%%% METRICS %%%%%%%%%%%%%%%%%%%%%
#%% Calculate volume for diastolic phase
#test_index = data_gt_ed[num_eval:num_test]

test_index = data_gt_ed[num_eval:num_test]

s = 0
target_vol_ed = np.zeros(len(test_index))
ref_vol_ed    = np.zeros(len(test_index))

for i in range(0,len(test_index)):
    for j in range(0, test_index[i].shape[0]):
        target_vol_ed[i] += np.sum(seg_dia[j+s,:,:,3])
        ref_vol_ed[i]    += np.sum(ref_dia[j+s,:,:,3])
        
    s += test_index[i].shape[0] 
   
#%% Calculate volume for systolic phase
test_index = data_gt_es[num_eval:num_test]

s = 0
target_vol_es = np.zeros(len(test_index))
ref_vol_es = np.zeros(len(test_index))

for i in range(0,len(test_index)):
    for j in range(0, test_index[i].shape[0]):
        target_vol_es[i] += np.sum(seg_sys[j+s,:,:,3])
        ref_vol_es[i]    += np.sum(ref_sys[j+s,:,:,3])
        
    s += test_index[i].shape[0] 
     
#%% Calculate EF        
#os.chdir('/Users/michalablicher/Documents/GitHub/Speciale2021')
os.chdir("C:/Users/katrine/Documents/GitHub/Speciale2021")

from metrics import EF_calculation, dc, hd, jc, precision, mcc, recall, risk, sensitivity, specificity, true_negative_rate, true_positive_rate, positive_predictive_value, hd95, assd, asd, ravd, volume_correlation, volume_change_correlation, obj_assd, obj_asd, obj_fpr, obj_tpr

#%%
spacings = [1.4, 1.4, 8]

ef_ref    = EF_calculation(ref_vol_es, ref_vol_ed, spacings)
ef_target = EF_calculation(target_vol_es, target_vol_ed, spacings)


ef_m_ref = np.mean(ef_ref[0])
ef_m_tar = np.mean(ef_target[0])

print('ef  = ', ef_ref[0]) 
print('esv = ', ef_ref[1]) 
print('edv = ', ef_ref[2]) 

print('ef  = ', ef_target[0]) 
print('esv = ', ef_target[1]) 
print('edv = ', ef_target[2]) 

#%%%%%%%%%%%%%% RESULTS DIASTOLIC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%  Caluclate Dice + Hausdorff 
dice_dia = np.zeros((seg_met_dia.shape[0],3))
haus_dia = np.zeros((seg_met_dia.shape[0],3))

# OBS OBS OBS OBS
# dim[0] = BG
# dim[1] = RV
# dim[2] = MYO
# dim[3] = LV

for i in range(0,seg_met_dia.shape[0]):
      
    dice_dia[i,0] = dc(seg_dia[i,:,:,1],ref_dia[i,:,:,1])  # = RV
    dice_dia[i,1] = dc(seg_dia[i,:,:,2],ref_dia[i,:,:,2])  # = MYO
    dice_dia[i,2] = dc(seg_dia[i,:,:,3],ref_dia[i,:,:,3])  # = LV
    
    # If there is no prediction or annotation then don't calculate Hausdorff distance and
    # skip to calculation for next class
    h_count = 0
    
    if len(np.unique(ref_dia[i,:,:,1]))!=1 and len(np.unique(seg_dia[i,:,:,1]))!=1:
        haus_dia[i,0]    = hd(seg_dia[i,:,:,1],ref_dia[i,:,:,1])  
        h_count += 1
    else:
        pass
    
    if len(np.unique(ref_dia[i,:,:,2]))!=1 and len(np.unique(seg_dia[i,:,:,2]))!=1:      
        haus_dia[i,1]    = hd(seg_dia[i,:,:,2],ref_dia[i,:,:,2])  
        h_count += 1
    else:
        pass
    
    if len(np.unique(ref_dia[i,:,:,3]))!=1 and len(np.unique(seg_dia[i,:,:,3]))!=1:
        haus_dia[i,2]    = hd(seg_dia[i,:,:,3],ref_dia[i,:,:,3])  
        h_count += 1
    else:
        pass
    
        pass        
    if h_count!= 3:
        print('Haus not calculated for all classes for slice: ', i)
    else:
        pass 
    
mean_dice_dia = np.mean(dice_dia, axis=0)  
std_dice_dia = np.std(dice_dia,  axis=0)

mean_haus_dia = np.mean(haus_dia, axis=0)
std_haus_dia = np.std(haus_dia,  axis=0)

print('mean dice = ',mean_dice_dia)  
print('std dice = ', std_dice_dia) 

print('mean haus = ',mean_haus_dia)
print('std haus = ', std_haus_dia)




#%% Calculate recall + precision

recall_dia    = np.zeros((seg_met_dia.shape[0],3))
precision_dia = np.zeros((seg_met_dia.shape[0],3))
for i in range(0,seg_met_dia.shape[0]):
      
    recall_dia[i,0] = recall(seg_dia[i,:,:,1],ref_dia[i,:,:,1])  # = RV
    recall_dia[i,1] = recall(seg_dia[i,:,:,2],ref_dia[i,:,:,2])  # = MYO
    recall_dia[i,2] = recall(seg_dia[i,:,:,3],ref_dia[i,:,:,3])  # = LV
    
    precision_dia[i,0] = precision(seg_dia[i,:,:,1],ref_dia[i,:,:,1])  # = RV
    precision_dia[i,1] = precision(seg_dia[i,:,:,2],ref_dia[i,:,:,2])  # = MYO
    precision_dia[i,2] = precision(seg_dia[i,:,:,3],ref_dia[i,:,:,3])  # = LV

mean_rec = np.mean(recall_dia, axis=0)  
mean_prec = np.mean(precision_dia, axis=0)
print('mean recall = ',mean_rec)  
print('mean precision = ',mean_prec)

#%% F1 score
F1_dia = 2 * ((precision_dia * recall_dia) / (precision_dia + recall_dia))    
mean_F1_dia = np.nanmean(F1_dia, axis=0) 

print('mean F1 = ',mean_F1_dia)  

#%% Mathew Correlation
mcc_dia    = np.zeros((seg_met_dia.shape[0],3))

for i in range(0,seg_met_dia.shape[0]):
    #mcc_dia[i,0] = mcc(seg_dia,ref_dia)
    mcc_dia[i,0] = mcc(seg_dia[i,:,:,1],ref_dia[i,:,:,1])  # = RV
    mcc_dia[i,1] = mcc(seg_dia[i,:,:,2],ref_dia[i,:,:,2])  # = MYO
    mcc_dia[i,2] = mcc(seg_dia[i,:,:,3],ref_dia[i,:,:,3])  # = LV
    
mean_mcc = np.mean(mcc_dia, axis=0)  
print('MCC = ',mean_mcc)  


#%% Calculate sensitivity + specificity
sensitivity_dia    = np.zeros((seg_met_dia.shape[0],3))
specificity_dia = np.zeros((seg_met_dia.shape[0],3))
for i in range(0,seg_met_dia.shape[0]):
      
    sensitivity_dia[i,0] = sensitivity(seg_dia[i,:,:,1],ref_dia[i,:,:,1])  # = RV
    sensitivity_dia[i,1] = sensitivity(seg_dia[i,:,:,2],ref_dia[i,:,:,2])  # = MYO
    sensitivity_dia[i,2] = sensitivity(seg_dia[i,:,:,3],ref_dia[i,:,:,3])  # = LV
    
    specificity_dia[i,0] = specificity(seg_dia[i,:,:,1],ref_dia[i,:,:,1])  # = RV
    specificity_dia[i,1] = specificity(seg_dia[i,:,:,2],ref_dia[i,:,:,2])  # = MYO
    specificity_dia[i,2] = specificity(seg_dia[i,:,:,3],ref_dia[i,:,:,3])  # = LV

mean_sensitivity = np.mean(sensitivity_dia, axis=0)  
mean_specificity = np.mean(specificity_dia, axis=0)
print('mean sensitivity = ',mean_sensitivity)  
print('mean specificity = ',mean_specificity)


#%%%%%%%%%%%%%% RESULTS SYSTOLIC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%  Caluclate Dice + Hausdorff 
dice_sys = np.zeros((seg_met_sys.shape[0],3))
haus_sys = np.zeros((seg_met_sys.shape[0],3))

# OBS OBS OBS OBS
# dim[0] = BG
# dim[1] = RV
# dim[2] = MYO
# dim[3] = LV

for i in range(0,seg_met_sys.shape[0]):
      
    dice_sys[i,0] = dc(seg_sys[i,:,:,1],ref_sys[i,:,:,1])  # = RV
    dice_sys[i,1] = dc(seg_sys[i,:,:,2],ref_sys[i,:,:,2])  # = MYO
    dice_sys[i,2] = dc(seg_sys[i,:,:,3],ref_sys[i,:,:,3])  # = LV
    
    # If there is no prediction or annotation then don't calculate Hausdorff distance and
    # skip to calculation for next class
    h_count = 0
    
    if len(np.unique(ref_sys[i,:,:,1]))!=1 and len(np.unique(seg_sys[i,:,:,1]))!=1:
        haus_sys[i,0]    = hd(seg_sys[i,:,:,1],ref_sys[i,:,:,1])  
        h_count += 1
    else:
        pass
    
    if len(np.unique(ref_sys[i,:,:,2]))!=1 and len(np.unique(seg_sys[i,:,:,2]))!=1:      
        haus_sys[i,1]    = hd(seg_sys[i,:,:,2],ref_sys[i,:,:,2])  
        h_count += 1
    else:
        pass
    
    if len(np.unique(ref_sys[i,:,:,3]))!=1 and len(np.unique(seg_sys[i,:,:,3]))!=1:
        haus_sys[i,2]    = hd(seg_sys[i,:,:,3],ref_sys[i,:,:,3])  
        h_count += 1
    else:
        pass
    
        pass        
    if h_count!= 3:
        print('Haus not calculated for all classes for slice: ', i)
    else:
        pass 
    
mean_dice_sys = np.mean(dice_sys, axis=0)  
mean_haus_sys = np.mean(haus_sys, axis=0)
print('mean dice = ',mean_dice_sys)  
print('mean haus = ',mean_haus_sys)

#%% Calculate recall + precision

recall_sys    = np.zeros((seg_met_sys.shape[0],3))
precision_sys = np.zeros((seg_met_sys.shape[0],3))
for i in range(0,seg_met_sys.shape[0]):
      
    recall_sys[i,0] = recall(seg_sys[i,:,:,1],ref_sys[i,:,:,1])  # = RV
    recall_sys[i,1] = recall(seg_sys[i,:,:,2],ref_sys[i,:,:,2])  # = MYO
    recall_sys[i,2] = recall(seg_sys[i,:,:,3],ref_sys[i,:,:,3])  # = LV
    
    precision_sys[i,0] = precision(seg_sys[i,:,:,1],ref_sys[i,:,:,1])  # = RV
    precision_sys[i,1] = precision(seg_sys[i,:,:,2],ref_sys[i,:,:,2])  # = MYO
    precision_sys[i,2] = precision(seg_sys[i,:,:,3],ref_sys[i,:,:,3])  # = LV

mean_rec = np.mean(recall_sys, axis=0)  
mean_prec = np.mean(precision_sys, axis=0)
print('mean recall = ',mean_rec)  
print('mean precision = ',mean_prec)

#%% F1 score
F1_sys = 2 * ((precision_sys * recall_sys) / (precision_sys + recall_sys))    
mean_F1_sys = np.nanmean(F1_sys, axis=0) 

print('mean F1 = ',mean_F1_sys)  

#%% Calculate sensitivity + specificity
sensitivity_sys    = np.zeros((seg_met_sys.shape[0],3))
specificity_sys = np.zeros((seg_met_sys.shape[0],3))
for i in range(0,seg_met_sys.shape[0]):
      
    sensitivity_sys[i,0] = sensitivity(seg_sys[i,:,:,1],ref_sys[i,:,:,1])  # = RV
    sensitivity_sys[i,1] = sensitivity(seg_sys[i,:,:,2],ref_sys[i,:,:,2])  # = MYO
    sensitivity_sys[i,2] = sensitivity(seg_sys[i,:,:,3],ref_sys[i,:,:,3])  # = LV
    
    specificity_sys[i,0] = specificity(seg_sys[i,:,:,1],ref_sys[i,:,:,1])  # = RV
    specificity_sys[i,1] = specificity(seg_sys[i,:,:,2],ref_sys[i,:,:,2])  # = MYO
    specificity_sys[i,2] = specificity(seg_sys[i,:,:,3],ref_sys[i,:,:,3])  # = LV

mean_sensitivity = np.mean(sensitivity_sys, axis=0)  
mean_specificity = np.mean(specificity_sys, axis=0)
print('mean sensitivity = ',mean_sensitivity)  
print('mean specificity = ',mean_specificity)


#%% Mathew Correlation
mcc_sys    = np.zeros((seg_met_sys.shape[0],3))

for i in range(0,seg_met_dia.shape[0]):
    #mcc_dia[i,0] = mcc(seg_dia,ref_dia)
    mcc_sys[i,0] = mcc(seg_sys[i,:,:,1],ref_sys[i,:,:,1])  # = RV
    mcc_sys[i,1] = mcc(seg_sys[i,:,:,2],ref_sys[i,:,:,2])  # = MYO
    mcc_sys[i,2] = mcc(seg_sys[i,:,:,3],ref_sys[i,:,:,3])  # = LV
    
mean_mcc = np.mean(mcc_sys, axis=0)  
print('MCC = ',mean_mcc)  




