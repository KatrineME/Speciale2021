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

y = gt_test_ed_sub[2,:,:]

Y = torch.nn.functional.one_hot(Tensor(y).to(torch.int64), num_classes=4).detach().numpy()


Y_BGR = Y[:,:,0]
Y_RV  = Y[:,:,1]
Y_MYO = Y[:,:,2]
Y_LV  = Y[:,:,3]

Y_LVmod = Y_LV
# Modify a GT to include an error where LV is in contact with BGR
Y_LVmod[50:56,20:39] = 1
# Modify a GT to include an error where LV is in contact with RV
Y_LVmod[15:33,50:55] = 1


H = 128
W = 128

# Shift
# Top, bottom, left,right
Y_LV_pad = np.pad(Y_LVmod,((1,1),(1,1)),'constant', constant_values=0)

Y_up   = Y_LV_pad[2:130,1:129]
Y_down = Y_LV_pad[0:128,1:129]

Y_left  = Y_LV_pad[1:129,2:130]
Y_right = Y_LV_pad[1:129,0:128]

#Y_UpLe = Y_LV_pad[2:130,2:130]
#Y_UpRi = Y_LV_pad[2:130,0:128]

#Y_DoRi = Y_LV_pad[0:128,0:128]
#Y_DoLe = Y_LV_pad[0:128,2:130]

#inside = (Y_up + Y_down + Y_left + Y_right + Y_UpLe + Y_UpRi + Y_DoRi + Y_DoLe) * (Y_BGR + Y_RV)
inside = (Y_up + Y_down + Y_left + Y_right) * (Y_BGR + Y_RV)
inside[inside > 0] = 1
#inside = ndimage.binary_erosion(inside).astype(inside.dtype)  # OBS: fjerner noget på inderside

loss = np.sum(inside) #/(128*128)
print('loss = ', loss)

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