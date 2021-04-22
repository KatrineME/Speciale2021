# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 11:55:30 2021

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
import re

#!pip install torch-summary
#!pip install opencv-python
#%%

cwd = os.getcwd()
os.chdir("C:/Users/katrine/Documents/Universitet/Speciale/ACDC_training_data/training")
#os.chdir('/Users/michalablicher/Desktop/training')

#frame_dia_im = np.sort(glob2.glob('patient*/**/patient*_frame01.nii.gz'))
frame_im = np.sort(glob2.glob('patient*/**/patient*_frame*[0-9].nii.gz'))
frame_gt = np.sort(glob2.glob('patient*/**/patient*_frame*[0-9]_gt.nii.gz'))

dia = np.linspace(0,len(frame_im)-2,100).astype(int)
sys = np.linspace(1,len(frame_im)-1,100).astype(int)

#%%
frame_dia_im = frame_im[dia]
frame_sys_im = frame_im[sys]

frame_dia_gt = frame_gt[dia]
frame_sys_gt = frame_gt[sys]

