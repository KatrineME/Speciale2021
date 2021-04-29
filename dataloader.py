#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 15:28:28 2021

@author: michalablicher
"""
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

#!pip install torch-summary
#!pip install opencv-python
from torch.utils.data import DataLoader


#%% Specify directory
#os.chdir("C:/Users/katrine/Documents/GitHub/Speciale2021")
os.chdir('/Users/michalablicher/Documents/GitHub/Speciale2021')

from load_data_gt_im import load_data

data_im_es, data_gt_es = load_data('M','Systole')
data_im_ed, data_gt_ed = load_data('M','Diastole')


#%% Load Data
num = 5

num_train = 50#num 
num_eval  = 50 + 30#num + num_train 
num_test  = 80 + 20#num + num_eval


im_flat_train = np.concatenate(data_im_ed[0:num_train]).astype(None)
gt_flat_train = np.concatenate(data_gt_ed[0:num_train]).astype(None)

im_flat_eval = np.concatenate(data_im_ed[num_train:num_eval]).astype(None)
gt_flat_eval = np.concatenate(data_gt_ed[num_train:num_eval]).astype(None)

im_flat_test = np.concatenate(data_im_ed[num_eval:num_test]).astype(None)
gt_flat_test = np.concatenate(data_gt_ed[num_eval:num_test]).astype(None)

#%%

data_train = Tensor((np.squeeze(im_flat_train), gt_flat_train))

train_dataloader = DataLoader(data_train, batch_size=2, shuffle=False)
eval_dataloader = DataLoader((np.squeeze(im_flat_eval), gt_flat_eval), batch_size=10, shuffle=True)

im_train , lab_train = next(iter(train_dataloader))
im_eval , lab_eval   = next(iter(eval_dataloader))
