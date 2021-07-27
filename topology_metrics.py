#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 16:39:22 2021

@author: michalablicher
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 15:29:54 2021

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
import skimage
from skimage import measure

#from skimage.transform import resize
from torch import nn
from torch import Tensor
import scipy.ndimage
import scipy.stats
import torchsummary


if torch.cuda.is_available():
    # Tensor = torch.cuda.FloatTensor
    device = 'cuda'
else:
    # Tensor = torch.FloatTensor
    device = 'cpu'
torch.cuda.manual_seed_all(808)

#%% BayesUNet
# recursive implementation of Unet

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
    
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.zero_()


class UNet(nn.Module):
    def __init__(self, num_classes=3, in_channels=1, initial_filter_size=64, kernel_size=3, num_downs=4,
                 norm_layer=nn.InstanceNorm2d, drop_prob=0.):
        super(UNet, self).__init__()
        self.drop_prob = drop_prob
        # construct UNet structure
        unet_block = UnetSkipConnectionBlock(in_channels=initial_filter_size * 2 ** (num_downs-1), out_channels=initial_filter_size * 2 ** num_downs,
                                             num_classes=num_classes, kernel_size=kernel_size, norm_layer=norm_layer,
                                             innermost=True, drop_prob=self.drop_prob)
        for i in range(1, num_downs):
            unet_block = UnetSkipConnectionBlock(in_channels=initial_filter_size * 2 ** (num_downs-(i+1)),
                                                 out_channels=initial_filter_size * 2 ** (num_downs-i),
                                                 num_classes=num_classes, kernel_size=kernel_size, submodule=unet_block,
                                                 norm_layer=norm_layer, drop_prob=self.drop_prob
                                                 )
            
        unet_block = UnetSkipConnectionBlock(in_channels=in_channels, out_channels=initial_filter_size,
                                             num_classes=num_classes, kernel_size=kernel_size, submodule=unet_block, norm_layer=norm_layer,
                                             outermost=True, drop_prob=self.drop_prob)

        self.model = unet_block
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.apply(weights_init)

    def forward(self, x):
        out = self.model(x)
        return {'log_softmax': self.log_softmax(out), 'softmax': self.softmax(out)}


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, num_classes=1, kernel_size=3,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.InstanceNorm2d, drop_prob=0.):
        super(UnetSkipConnectionBlock, self).__init__()
        self.use_dropout = True if drop_prob > 0. else False
        self.drop_prob = drop_prob
        self.outermost = outermost
        # downconv
        pool = nn.MaxPool2d(2, stride=2)
        conv1 = self.contract(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, norm_layer=norm_layer)
        conv2 = self.contract(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, norm_layer=norm_layer)

        # upconv
        conv3 = self.expand(in_channels=out_channels*2, out_channels=out_channels, kernel_size=kernel_size)
        conv4 = self.expand(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size)

        if outermost:
            final = nn.Conv2d(out_channels, num_classes, kernel_size=1)
            if self.use_dropout:
                down = [conv1, conv2, nn.Dropout2d(self.drop_prob)]
            else:
                down = [conv1, conv2]
            if self.use_dropout:
                up = [conv3, nn.Dropout2d(self.drop_prob), conv4, nn.Dropout2d(self.drop_prob), final]
            else:
                up = [conv3, conv4, final]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(in_channels*2, in_channels,
                                        kernel_size=2, stride=2)
            model = [pool, conv1, conv2, upconv]
        else:
            upconv = nn.ConvTranspose2d(in_channels*2, in_channels, kernel_size=2, stride=2)

            down = [pool, conv1, conv2]
            up = [conv3, conv4, upconv]

            if self.use_dropout:
                model = down + [nn.Dropout2d(self.drop_prob)] + [submodule] + up + [nn.Dropout2d(self.drop_prob)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    @staticmethod
    def contract(in_channels, out_channels, kernel_size=3, norm_layer=nn.InstanceNorm2d):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            norm_layer(out_channels),
            nn.LeakyReLU(inplace=True))
        return layer

    @staticmethod
    def expand(in_channels, out_channels, kernel_size=3):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        return layer

    @staticmethod
    def center_crop(layer, target_width, target_height):
        batch_size, n_channels, layer_width, layer_height = layer.size()

        xy1 = (layer_width - target_width) // 2
        xy2 = (layer_height - target_height) // 2
        return layer[:, :, xy1:(xy1 + target_width), xy2:(xy2 + target_height)]

    def forward(self, x):
        if self.outermost:
            out = self.model(x)
            return out
        else:
            crop = self.center_crop(self.model(x), x.size()[2], x.size()[3])
            out = torch.cat([x, crop], 1)
            return out


class BayesUNet(UNet):

    def __init__(self, num_classes=3, in_channels=1, initial_filter_size=64, kernel_size=3, num_downs=4,
                 norm_layer=nn.InstanceNorm2d, drop_prob=0.):
        super(BayesUNet, self).__init__(num_classes, in_channels, initial_filter_size, kernel_size, num_downs,
                 norm_layer=norm_layer, drop_prob=drop_prob)

    def train(self, mode=True, mc_dropout=False):
        """ Sets the module in training mode.
            !!! OVERWRITING STANDARD PYTORCH METHOD for nn.Module

            OR
                if mc_dropout=True and mode=False (use dropout during inference) we set all modules
                to train-mode=False except for DROPOUT layers
                In this case it is important that the module_name matches BayesDRNSeg.dropout_layer

        Returns:
            Module: self
        """
        self.training = mode
        for module_name, module in self.named_modules():
            module.training = mode
            if mc_dropout and not mode:
                if isinstance(module, nn.Dropout2d):
                    # print("WARNING - nn.Module.train - {}".format(module_name))
                    module.training = True

        return self

    def eval(self, mc_dropout=False):
        """Sets the module in evaluation mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.
        """
        return self.train(False, mc_dropout=mc_dropout)

if __name__ == "__main__":
    #import torchsummary
    unet = BayesUNet(num_classes=4, in_channels=1, drop_prob=0.1)
    
    if device == 'cuda':
        unet.cuda()
    #torchsummary.summary(unet, (1, 128, 128))

#%% Specify directory
if device == 'cuda':
    user = 'GPU'
else:
    user = 'K'

user = 'K'
if user == 'M':
    os.chdir('/Users/michalablicher/Documents/GitHub/Speciale2021')
if user == 'K':
    os.chdir('C:/Users/katrine/Documents/GitHub/Speciale2021')
if user == 'GPU':
    os.chdir('/home/michala/Speciale2021/Speciale2021')

 
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
"""
im_train_ed_sub = np.concatenate((np.concatenate(data_im_ed_DCM[0:num_train_sub]).astype(None),
                                  np.concatenate(data_im_ed_HCM[0:num_train_sub]).astype(None),
                                  np.concatenate(data_im_ed_MINF[0:num_train_sub]).astype(None),
                                  np.concatenate(data_im_ed_NOR[0:num_train_sub]).astype(None),
                                  np.concatenate(data_im_ed_RV[0:num_train_sub]).astype(None)))

gt_train_ed_sub = np.concatenate((np.concatenate(data_gt_ed_DCM[0:num_train_sub]).astype(None),
                                  np.concatenate(data_gt_ed_HCM[0:num_train_sub]).astype(None),
                                  np.concatenate(data_gt_ed_MINF[0:num_train_sub]).astype(None),
                                  np.concatenate(data_gt_ed_NOR[0:num_train_sub]).astype(None),
                                  np.concatenate(data_gt_ed_RV[0:num_train_sub]).astype(None)))

gt_test_ed_sub = gt_train_ed_sub
im_test_ed_sub = im_train_ed_sub
"""
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

print('Data loaded+concat')

#%% Load model if averagered on GPU

#path_out_soft = '/Users/michalablicher/Desktop/Out_softmax_fold_avg_150dia_CE.pt'
path_out_soft = 'C:/Users/katrine/Desktop/Optuna/Final CV models/Out_softmax_fold_avg_150sys_dice_lclv_opt.pt'

out_soft = torch.load(path_out_soft ,  map_location=torch.device(device))

#%% Mean + argmax + one hot

out_soft_mean   = out_soft.mean(axis=0)

#out_soft_mean   = out_soft[5,:,:,:,:]
out_seg_mean_am = np.argmax(out_soft_mean, axis=1)
out_seg_mean    = torch.nn.functional.one_hot(torch.as_tensor(out_seg_mean_am), num_classes=4).detach().cpu().numpy()

ref = torch.nn.functional.one_hot(torch.as_tensor(Tensor(gt_test_es_sub).to(torch.int64)), num_classes=4).detach().cpu().numpy()

#%%
def lv_loss(y_pred):
    Y_BGR  = y_pred[:,0,:,:]           # size([B,H,W])
    Y_RV   = y_pred[:,1,:,:]           # size([B,H,W])
    Y_LV   = y_pred[:,3,:,:]           # size([B,H,W])

    Y_LV_pad = torch.nn.functional.pad(Y_LV,(1,1,1,1),'constant', 0)

    Y_up   = Y_LV_pad[:,2:130,1:129]
    Y_down = Y_LV_pad[:,0:128,1:129]
    
    Y_left = Y_LV_pad[:,1:129,2:130]
    Y_right= Y_LV_pad[:,1:129,0:128]
    

    inside = (Y_up + Y_down + Y_left + Y_right) * (Y_BGR + Y_RV)
    inside = inside.detach().cpu()#cuda()

    #print('inside', inside)    
    return inside # torch.sum(Tensor(inside))/(128*128*32)#.cuda()

out_seg_per = Tensor(out_seg_mean).permute(0,3,1,2)
lv_neigh = lv_loss(out_seg_per)

c_non = np.count_nonzero(lv_neigh, axis = (1,2)) # number of error pixels in each slice

cnon_slice = np.count_nonzero(c_non) # number of slices with erros 
print('Number of LV neighbouring pixels: ', np.sum(c_non))
print('Number of slices with errors:', cnon_slice)
print('Percentage of slices with errors:', (cnon_slice/len(c_non))*100,'%')
#print('Number of errornous neighbour pixels:', c_non.sum())
print('\n')

# Slices per patient
p = []

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
    
test_index = len(p)

s = 0
cnon_pt = np.zeros(test_index)

for i in range(0,test_index):
    for j in range(0, p[i]):
        cnon_pt[i] += np.count_nonzero(c_non[j+s])
        
    s += p[i] 
    #print('s= ',s)
print('Number of patient volumes w. errors:',np.count_nonzero(cnon_pt))
print('Percentage of patient volumes w. errors:',(np.count_nonzero(cnon_pt)/len(p))*100)   


#%%
labeled_image_rv = []
labeled_image_myo = []
labeled_image_lv = []
labeled_image_all = []

out_seg_mean_bin = (out_seg_mean_am > 0).astype(int)

for i in range(0, (out_seg_mean.shape[0])):
    labeled_image_all.append(skimage.measure.label(out_seg_mean_bin[i,:,:], connectivity=2, return_num=True))
    labeled_image_rv.append(skimage.measure.label(out_seg_mean[i,:,:,1], connectivity=2, return_num=True))
    labeled_image_myo.append(skimage.measure.label(out_seg_mean[i,:,:,2], connectivity=2, return_num=True))
    labeled_image_lv.append(skimage.measure.label(out_seg_mean[i,:,:,3], connectivity=2, return_num=True))

#%%
slice = 15
plt.figure(dpi=200)
plt.subplot(2,2,1)
plt.imshow(labeled_image_rv[slice][0])
plt.subplot(2,2,2)
plt.imshow(labeled_image_myo[slice][0])
plt.subplot(2,2,3)
plt.imshow(labeled_image_lv[slice][0])
plt.subplot(2,2,4)
plt.imshow(labeled_image_all[slice][0])

#%%
multi_lab_rv = []
multi_lab_myo = []
multi_lab_lv = []
multi_lab_all = []

for i in range(0, (out_seg_mean.shape[0])):
    multi_lab_rv.append(float(labeled_image_rv[i][1] > 1))
    multi_lab_myo.append(float(labeled_image_myo[i][1] > 1))
    multi_lab_lv.append(float(labeled_image_lv[i][1] > 1))
    multi_lab_all.append(float(labeled_image_all[i][1] > 1))


tot_disconnect = np.sum(multi_lab_all)
 

tot_rv = np.sum(multi_lab_rv)
tot_myo = np.sum(multi_lab_myo)
tot_lv = np.sum(multi_lab_lv)

all_tot_sum = []
for i in range(0, (out_seg_mean.shape[0])):
    all_tot_sum.append(multi_lab_rv[i] + multi_lab_myo[i] + multi_lab_lv[i])


c_non_bin = (c_non > 0).astype(int)
tot_miss_bin = (np.array(all_tot_sum) > 0).astype(int)


final_both = tot_miss_bin + c_non_bin# + multi_lab_all
final_bin = (final_both > 0).astype(int)
final_count = np.sum(final_bin)

print('Number of slices w. multiple components:',np.count_nonzero(tot_miss_bin))
print('Percentage of slices w. multiple components:',(np.count_nonzero(tot_miss_bin)/len(tot_miss_bin))*100)   
print('\n')
print('Number of slices w. disconnect MYO RV:',np.count_nonzero(multi_lab_all))
print('Percentage of slices w. disconnect MYO RV:',(np.count_nonzero(multi_lab_all)/len(multi_lab_all))*100)  

#%%  Slices per patient
p = []

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
    
test_index = len(p)

s = 0
final_count_pt = np.zeros(test_index)
final_myo_rv_pt = np.zeros(test_index)

for i in range(0,test_index):
    for j in range(0, p[i]):
        final_count_pt[i] += np.count_nonzero(tot_miss_bin[j+s])
        final_myo_rv_pt[i] += np.count_nonzero(multi_lab_all[j+s])
        
    s += p[i] 
    #print('s= ',s)



print('Number of patient volumes w. multiple comp:',np.count_nonzero(final_count_pt))
print('Percentage of patient volumes w. multiple comp:',(np.count_nonzero(final_count_pt)/len(p))*100)   
print('\n')
print('Number of patient volumes w. disconnect MYO RV:',np.count_nonzero(final_myo_rv_pt))
print('Percentage of patient volumes w. disconnect MYO RV:',(np.count_nonzero(final_myo_rv_pt)/len(p))*100) 


#%% OVERALL

overall_errors_slices = (np.array(multi_lab_all) + final_both)
overall_errors_slices_bin = overall_errors_slices > 0
overall_errors_count = overall_errors_slices_bin.sum()

print('Number of slices w. any error:',overall_errors_count)
print('Percentage of slices w.any error:',overall_errors_count/len(overall_errors_slices)*100)  
print('\n')

s = 0
overall_count_pt = np.zeros(test_index)

for i in range(0,test_index):
    for j in range(0, p[i]):
        overall_count_pt[i] += np.count_nonzero(overall_errors_slices_bin[j+s])
        
    s += p[i] 
    #print('s= ',s)
print('Number of patient volumes w.any error:',np.count_nonzero(overall_count_pt))
print('Percentage of patient volumes w. any error:',(np.count_nonzero(overall_count_pt)/len(p))*100)   

#%% Plot of anantomical errors

alpha = 0.4

slice_myo = 28
slice_com = 34
slice_dis = 67
slice_nor = 56

plt.figure(dpi=400, figsize=(17,13))
plt.suptitle('Examples of anatomical errors', fontsize=25, y=0.95)

plt.subplot(3,4,1)
plt.imshow(out_seg_mean_am[slice_myo,:,:])
#plt.imshow(im_test_ed_sub[slice_myo,0,:,:], alpha = alpha)
plt.title('LV not fully surrounded by MYO', fontsize=17)
plt.ylabel('Segmentation', fontsize=17)
plt.subplot(3,4,1+4)
plt.imshow(gt_test_ed_sub[slice_myo,:,:])
plt.ylabel('Reference', fontsize=17)
plt.subplot(3,4,1+8)
plt.imshow(im_test_ed_sub[slice_myo,0,:,:])
plt.ylabel('Original cMRI', fontsize=17)


plt.subplot(3,4,2)
plt.imshow(out_seg_mean_am[slice_com,:,:])
#plt.imshow(im_test_ed_sub[slice_com,0,:,:], alpha = alpha)
plt.title('Not single components', fontsize=17)
plt.subplot(3,4,2+4)
plt.imshow(gt_test_ed_sub[slice_com,:,:])
plt.subplot(3,4,2+8)
plt.imshow(im_test_ed_sub[slice_com,0,:,:])


plt.subplot(3,4,3)
plt.imshow(out_seg_mean_am[slice_dis,:,:])
#plt.imshow(im_test_ed_sub[slice_dis,0,:,:], alpha = alpha)
plt.title('Disconnected components', fontsize=17)
plt.subplot(3,4,3+4)
plt.imshow(gt_test_ed_sub[slice_dis,:,:])
plt.subplot(3,4,3+8)
plt.imshow(im_test_ed_sub[slice_dis,0,:,:])

plt.subplot(3,4,4)
plt.imshow(out_seg_mean_am[slice_nor,:,:])
#plt.imshow(im_test_ed_sub[slice_nor,0,:,:], alpha = alpha)
plt.title('No anatomical errors', fontsize=17)
plt.subplot(3,4,4+4)
plt.imshow(gt_test_ed_sub[slice_nor,:,:])
plt.subplot(3,4,4+8)
plt.imshow(im_test_ed_sub[slice_nor,0,:,:])
