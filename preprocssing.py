# -*- coding: utf-8 -*-
"""
Created on Thu May 13 11:30:28 2021

@author: katrine
"""

import os
import cv2
import glob2
import torchvision

from torch import Tensor
from PIL   import Image

import nibabel as nib
import numpy   as np
from torch import nn
import torch
import matplotlib.pyplot as plt
#%%
    
os.chdir("C:/Users/katrine/Documents/Universitet/Speciale/ACDC_training_data/training")

phase = 'Diastole'
frame_im = np.sort(glob2.glob('patient*/**/patient*_frame*[0-9].nii.gz'))
frame_gt = np.sort(glob2.glob('patient*/**/patient*_frame*[0-9]_gt.nii.gz'))


if phase == 'Diastole':
   phase = np.linspace(0,len(frame_im)-2,100).astype(int)
else:
   phase = np.linspace(1,len(frame_im)-1,100).astype(int)
        
# Divide frames
frame_im = frame_im[phase]
frame_gt = frame_gt[phase]

H = 128
W = H

centercrop = torchvision.transforms.CenterCrop((H,W))

num_slices = np.zeros(100)

im_data = []
gt_data = [] 
circles = []
ori_resol = np.zeros((100,2))

for i in range(0,100):
    nimg = nib.load(frame_im[i])
    img  = nimg.get_fdata()
    
    n_gt = nib.load(frame_gt[i])
    gt   = n_gt.get_fdata()
    
    im_slices = img.shape[2]
    
    num_slices[i] = im_slices
    
    ori_resol[i,0] = img.shape[0]
    ori_resol[i,1] = img.shape[1]
    
    crop_off = (ori_resol-H)/2
    
    centercrop_img = Tensor(np.zeros((H,W,im_slices)))
    crop_gt  = np.zeros((H,W,im_slices))
                            
    for j in range(0,im_slices):
        centercrop_img[:,:,j] = centercrop(Tensor(img[:,:,j]))
    
    in_image = np.expand_dims(centercrop_img,0)
    in_image = Tensor(in_image).permute(3,0,1,2).detach().numpy()
    
    img_center1 = in_image[int(im_slices/2),0,:,:] 
    
    img_center = cv2.normalize(img_center1, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    circ = cv2.HoughCircles(img_center, cv2.HOUGH_GRADIENT, 1, img_center.shape[0], param1=200, param2=10, minRadius=5, maxRadius=35)
    circles.append(circ)
    
    #print(i)
    
    left  = int((circles[i][0][0][0]+crop_off[i][1])-(H/2))
    upper = int((circles[i][0][0][1]+crop_off[i][0])-(H/2))
    right = int((circles[i][0][0][0]+crop_off[i][1])+(H/2))
    lower = int((circles[i][0][0][1]+crop_off[i][0])+(H/2))
    
    #print(left, upper, right, lower)
    #print(gt.shape)
    """
    for j in range(0,im_slices):
        gt_pt = gt[:,:,j]
        
        print('before right', gt_pt.shape)
        if gt_pt.shape[1] < right:
            gt_pt = np.pad(gt_pt,((0,0),(0,0)))
        
        print('before lower', gt_pt.shape)
        if gt_pt.shape[0] < right:
            gt_pt = np.pad(gt_pt,((0,right-gt_pt.shape[0]),(0,0)))
        
        print('before left', gt_pt.shape)
        if left == 0:
            right+=0
            
        print('after right', gt_pt.shape)
        
        crop_gt[:,:,j]  = gt_pt[left:right, upper:lower]
"""
    im_data.append(in_image.astype(object))
    
    in_gt = Tensor(crop_gt).permute(2,0,1).detach().numpy()
    gt_data.append(in_gt.astype(object))
#%%
im = np.concatenate(im_data).astype(None)
gt = np.concatenate(gt_data).astype(None)

ref_dia = torch.nn.functional.one_hot(Tensor(gt).to(torch.int64), num_classes=4).detach().numpy()
s = np.sum(ref_dia,(0,1,2))

#%% Center slices
center_slices = np.zeros(100)
slices = 0

for i in range(0,100):
    cent = int(im_data[i].shape[0]/2)    
    center_slices[i] = int(im_data[i].shape[0]/2+slices)
    slices += im_data[i].shape[0]

center_slices.astype(np.int64)


#patient = 99  # fail
patient = 0 # correct

nimg = nib.load(frame_im[patient])
img  = nimg.get_fdata()

s = int(img.shape[2]/2)
image = img[:,:,s]

crop_off = (ori_resol-H)/2

fig, ax = plt.subplots(dpi=200)
plt.imshow(image)

for i in range(0,circles[patient].shape[1]):
    x = circles[patient][i][0][0]+crop_off[patient][1]
    y = circles[patient][i][0][1]+crop_off[patient][0]
    ci = plt.Circle((x,y), circles[patient][i][0][2], color='red', alpha=0.4)
    ax.add_patch(ci)


#%% Working plot
s = int(center_slices[patient])
image = im[s,0,:,:]

fig, ax = plt.subplots()
plt.imshow(image)
for i in range(0,circles[patient].shape[1]):
    c = plt.Circle((circles[patient][i][0][0],circles[patient][i][0][1]), circles[patient][i][0][2], color='red', alpha=0.4)
    ax.add_patch(c)

#param1=200, param2=10
edges = cv2.Canny(np.uint8(image),200,10)

plt.imshow(edges)
plt.title('Canny for patient 1')

#%%

# plot circles 
patient = 3

print('Number of circles: ',circles[patient][0].shape[0])
#print('Number of multi detect:',np.count_nonzero(circles[:][0].shape[0] > 1))

nimg = nib.load(frame_im[patient])
img  = nimg.get_fdata()

slice = int(img.shape[2]/2)

fig, ax = plt.subplots()
#for i in range(0,im_c.shape[0]):
#    plt.subplot(2,5,i+1)
plt.imshow(img[:,:,5])
circle1 = plt.Circle((circles[patient][0][0][0],circles[patient][0][0][1]), circles[patient][0][0][2], color='red', alpha=0.2)
ax.add_patch(circle1)



#%%
patient = 37
nimg = nib.load(frame_im[patient])
img  = nimg.get_fdata()
img = img[:,:,4]

plt.figure(dpi=200)
plt.subplot(1,2,1)
plt.imshow(img)

#img_blur = (255*(img - np.min(img))/np.ptp(img)).astype(int)
#img   = cv2.imread("C:/Users/katrine/Documents/Universitet/Speciale/eyes.jpg")
#gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = cv2.normalize(img, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
#gray = cv2.CV_8UC1(img_blur)

gray = np.array(gray)
plt.subplot(1,2,2)
plt.imshow(img,cmap='gray')

gray = cv2.normalize(img, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, gray.shape[0], param1=200, param2=10, minRadius=5, maxRadius=35)
print(circles.shape)
print(circles)

fig, ax = plt.subplots(dpi=200)
plt.imshow(gray)
for i in range(0,circles.shape[1]):
    c = plt.Circle((circles[0][i][0],circles[0][i][1]), circles[0][i][2], color='red', alpha=0.2)
    ax.add_patch(c)

#%%


# Draw detected circles
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Draw outer circle
        cv2.circle(gray, (i[0], i[1]), i[2], 255, 2)
        # Draw inner circle
        cv2.circle(gray, (i[0], i[1]), 2, 90, 3)

#%%

slice = int(center_slices[37])
plt.figure(dpi=200)
plt.imshow(im[slice,0,:,:])