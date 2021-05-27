# -*- coding: utf-8 -*-
"""
Created on Tue May 25 12:23:01 2021

@author: katrine
"""

from torch import Tensor
import numpy   as np
import matplotlib.pyplot as plt
from scipy import ndimage

y = gt_train_sub[2,:,:]

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


# Shift
# Top, bottom, left,right
Y_LV_pad = np.pad(Y_LVmod,((1,1),(1,1)),'constant', constant_values=0)

Y_up   = Y_LV_pad[2:130,1:129]
Y_down = Y_LV_pad[0:128,1:129]

Y_left  = Y_LV_pad[1:129,2:130]
Y_right = Y_LV_pad[1:129,0:128]

Y_UpLe = Y_LV_pad[2:130,2:130]
Y_UpRi = Y_LV_pad[2:130,0:128]

Y_DoRi = Y_LV_pad[0:128,0:128]
Y_DoLe = Y_LV_pad[0:128,2:130]

inside = (Y_up + Y_down + Y_left + Y_right + Y_UpLe + Y_UpRi + Y_DoRi + Y_DoLe) * (Y_BGR + Y_RV)
inside[inside > 0] = 1
#inside = ndimage.binary_erosion(inside).astype(inside.dtype)  # OBS: fjerner noget på inderside

loss = np.sum(inside)
print('loss = ', loss)

#plt.figure(dpi=200)
#plt.imshow(inside)
#plt.title('Pixels penalized for neighbourhood')

plt.figure(dpi=200)
plt.imshow(Y_LVmod)
plt.imshow(Y_MYO, alpha=0.2)
plt.imshow(Y_RV, alpha=0.2)
plt.imshow(inside, alpha=0.2)
plt.title('Pixels penalized for neighbourhood')

#%%
plt.figure(dpi=200)
plt.imshow(inside)
plt.imshow(Y_MYO, alpha=0.2)
plt.imshow(Y_LVmod, alpha=0.2)
plt.imshow(Y_RV, alpha=0.2)
plt.title('Pixels penalized for neighbourhood')