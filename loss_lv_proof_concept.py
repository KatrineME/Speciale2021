# -*- coding: utf-8 -*-
"""
Created on Tue May 25 12:23:01 2021

@author: katrine
"""

from torch import Tensor
import numpy   as np
import matplotlib.pyplot as plt
from scipy import ndimage
import torch
import os
import nibabel as nib
import numpy   as np
import torchvision
import glob2
import torch.optim as optim
from scipy import ndimage

from torch.autograd  import Variable
from torch import nn
from torch import Tensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
#%%

#y = gt_test_ed_sub[2,:,:]

device = 'cpu'

#path_out_soft = '/Users/michalablicher/Desktop/Out_softmax_fold_avg_200dia_dice_10lclv.pt'
path_out_soft = 'C:/Users/katrine/Desktop/Optuna/Out_softmax_fold_avg_200dia_dice_2lclv.pt'

out_soft = torch.load(path_out_soft ,  map_location=torch.device(device))

out_soft = out_soft.mean(axis=0)
y_pred = Tensor(out_soft).permute(0,2,3,1)

# Penalize bgr next to lv
poteUP    = y_pred[:,0:-1,:,0]*y_pred[:,1:,:,3]
poteDOWN  =  y_pred[:,1:,:,0] *y_pred[:,0:-1,:,3]
poteUD    = poteUP+poteDOWN
poteUD    = torch.mean(poteUD, axis=1)
poteUD    = torch.mean(poteUD, axis=1)

poteLEFT  = y_pred[:,:,1:,0]  *y_pred[:,:,0:-1,3]
poteRIGHT = y_pred[:,:,0:-1,0]*y_pred[:,:,1:,3]
poteLR    = poteLEFT+poteRIGHT
poteLR    = torch.mean(poteLR, axis=1)
poteLR    = torch.mean(poteLR, axis=1)

pote1      = poteUD + poteLR
#%%
# Penalize rv next to lv
poteUP   = y_pred[:,0:-1,:,1]*y_pred[:,1:,:,3]
poteDOWN =  y_pred[:,1:,:,1] *y_pred[:,0:-1,:,3]
poteUD   = poteUP+poteDOWN
poteUD   = torch.mean(poteUD, axis=1)
poteUD   = torch.mean(poteUD, axis=1)

poteLEFT  = y_pred[:,:,1:,1]   *y_pred[:,:,0:-1,3]
poteRIGHT = y_pred[:,:,0:-1,1] *y_pred[:,:,1:,3]
poteLR    = poteLEFT+poteRIGHT
poteLR    = torch.mean(poteLR, axis=1)
poteLR    = torch.mean(poteLR, axis=1)

pote2 = poteUD + poteLR


#%%
pote1 = 0.25*pote1 / 2
pote2 = 0.25*pote2 / 2
#%%
loss = torch.mean(pote1+pote2)
print(loss)
#loss with 0.25/2 => tensor(0.0004)
#loss else        => tensor(0.0034)

#%%
Y = y_pred

#%%

Y_BGR = Y[:,:,:,0]
Y_RV  = Y[:,:,:,1]
Y_MYO = Y[:,:,:,2]
Y_LV  = Y[:,:,:,3]

#Y_LVmod = Y_LV
# Modify a GT to include an error where LV is in contact with BGR
#Y_LVmod[50:56,20:39] = 0
# Modify a GT to include an error where LV is in contact with RV
#Y_LVmod[15:33,50:55] = 0

#Y_LVmod[10:20,20:25] = 1

H = 128
W = 128

# Shift
# Top, bottom, left,right
Y_LV_pad = torch.nn.functional.pad(Y_LV,(1,1,1,1),'constant',0)

Y_up   = Y_LV_pad[:,2:130,1:129]
Y_down = Y_LV_pad[:,0:128,1:129]

Y_left  = Y_LV_pad[:,1:129,2:130]
Y_right = Y_LV_pad[:,1:129,0:128]

#Y_UpLe = Y_LV_pad[2:130,2:130]
#Y_UpRi = Y_LV_pad[2:130,0:128]

#Y_DoRi = Y_LV_pad[0:128,0:128]
#Y_DoLe = Y_LV_pad[0:128,2:130]

#inside = (Y_up + Y_down + Y_left + Y_right + Y_UpLe + Y_UpRi + Y_DoRi + Y_DoLe) * (Y_BGR + Y_RV)
inside = (Y_up + Y_down + Y_left + Y_right) * (Y_BGR + Y_RV)
#inside[inside > 0] = 1
#inside = ndimage.binary_erosion(inside).astype(inside.dtype)  # OBS: fjerner noget på inderside

loss = torch.sum(inside)/(128*128*337)
print('loss = ', loss)
#%%
#plt.figure(dpi=200)
#plt.imshow(inside)
#plt.title('Pixels penalized for neighbourhood')

plt.figure(dpi=200)
plt.imshow(Y_LVmod)
plt.imshow(Y_MYO, alpha=0.2)
plt.imshow(Y_RV, alpha=0.2)
#plt.imshow(inside, alpha=0.2)
plt.title('Pixels penalized for neighbourhood')

#%%
plt.figure(dpi=200)
plt.imshow(inside)
plt.imshow(Y_MYO, alpha=0.2)
plt.imshow(Y_LVmod, alpha=0.2)
plt.imshow(Y_RV, alpha=0.2)
plt.title('Pixels penalized for neighbourhood')




#%% Class loss

c = 1
slice = 41

print('Slice = ', slice)

y_true = ref_sys_oh[slice:slice+1,:,:,:]
y_pred = seg_sys_oh[slice:slice+1,:,:,:]

plt.imshow(y_true[0,:,:,c])
plt.imshow(y_pred[0,:,:,c], alpha=0.6)



eps = 1e-6
y_true =Tensor(y_true)
y_pred =Tensor(y_pred)

y_true_s   = torch.sum(y_true, (1,2))
y_true_sin = torch.empty((y_true_s.shape))
    
y_true_sin[y_true_s > 0]  = 0
y_true_sin[y_true_s == 0] = 1
  
#y_pred_e = torch.exp(y_pred)
loss_c = -1* torch.sum(torch.log(1-y_pred + eps),(1,2))
print(loss_c)

loss_c = loss_c*y_true_sin
print(loss_c)

loss_c = torch.sum(loss_c)
print(loss_c)

loss_c = loss_c/(y_pred.shape[3]*y_pred.shape[2]*y_pred.shape[1]*y_pred.shape[0])

print(loss_c)
#%%
plt.figure(dpi=200)
plt.subplot(1,3,1)
plt.imshow(z1.detach().numpy())
plt.title('Input image (no RV)')

plt.subplot(1,3,2)
plt.imshow(x1[0,1,:,:].detach().numpy())
plt.title('Predicted RV')

plt.subplot(1,3,3)
plt.imshow(torch.argmax(y1[0,:,:,:],axis=0).detach().numpy())
plt.title('Annotation (no RV)')