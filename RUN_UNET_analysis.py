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
    #model.cuda()
    #torchsummary.summary(model, (1, 128, 128))

#%% Specify directory
#os.chdir('/Users/michalablicher/Documents/GitHub/Speciale2021')
os.chdir('C:/Users/katrine/Documents/GitHub/Speciale2021')
from load_data_gt_im_sub import load_data_sub
user = 'K'
"""
data_im_es_DCM,  data_gt_es_DCM  = load_data_sub('M','Systole','DCM')
data_im_es_HCM,  data_gt_es_HCM  = load_data_sub('M','Systole','HCM')
data_im_es_MINF, data_gt_es_MINF = load_data_sub('M','Systole','MINF')
data_im_es_NOR,  data_gt_es_NOR  = load_data_sub('M','Systole','NOR')
data_im_es_RV,   data_gt_es_RV   = load_data_sub('M','Systole','RV')
"""
data_im_ed_DCM,  data_gt_ed_DCM  = load_data_sub(user,'Diastole','DCM')
data_im_ed_HCM,  data_gt_ed_HCM  = load_data_sub(user,'Diastole','HCM')
data_im_ed_MINF, data_gt_ed_MINF = load_data_sub(user,'Diastole','MINF')
data_im_ed_NOR,  data_gt_ed_NOR  = load_data_sub(user,'Diastole','NOR')
data_im_ed_RV,   data_gt_ed_RV   = load_data_sub(user,'Diastole','RV')



#%% BATCH GENERATOR
num_train_sub = 12
num_eval_sub = num_train_sub
num_test_sub = num_eval_sub + 8

"""
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


#%% Load Model
#PATH_model = "C:/Users/katrine/Documents/GitHub/Speciale2021/trained_Unet_testtest.pt"
#PATH_state = "C:/Users/katrine/Documents/GitHub/Speciale2021/trained_Unet_testtestate.pt"

#PATH_model_es = '/Users/michalablicher/Desktop/Trained_Unet_CE_sys_sub_batch_100.pt'
PATH_model_ed = '/Users/michalablicher/Desktop/Trained_Unet_CE_dia_CrossVal_optuna.pt'

#%% Import results from training (Loss + Accuracy)
PATH_res_ed = '/Users/michalablicher/Desktop/Trained_Unet_dice_sys_100e_10lc10lv_train_results_scheduler.pt'
res_ed = torch.load(PATH_res_ed, map_location=torch.device('cpu'))

#%% Load model
#unet_es = torch.load(PATH_model_es, map_location=torch.device('cpu'))
unet_ed = torch.load(PATH_model_ed, map_location=torch.device('cpu'))

#%% U-net Systolic data
unet_es.eval()
out_trained_es = unet_es(Tensor(im_test_es_sub))
out_image_es    = out_trained_es["softmax"]

#%% U-net Diastolic data
unet_ed.eval()
out_trained_ed = unet_ed(Tensor(im_test_ed_sub))
out_image_ed    = out_trained_ed["softmax"]

#%% Argmax and one-hot encoding
#seg_met_dia = np.argmax(out_image_ed.detach().numpy(), axis=1)











#%% Load softmax from ensemble models
PATH_softmax_ensemble_unet = 'C:/Users/katrine/Desktop/Optuna/Out_softmax_fold_avg.pt'
out_softmax_unet_fold = torch.load(PATH_softmax_ensemble_unet ,  map_location=torch.device(device))

# mean them over dim=0
out_softmax_unet = out_softmax_unet_fold.mean(axis=0)
seg_met_dia = np.argmax(out_softmax_unet, axis=1)

seg_dia = torch.nn.functional.one_hot(torch.as_tensor(seg_met_dia), num_classes=4).detach().numpy()
ref_dia = torch.nn.functional.one_hot(Tensor(gt_test_ed_sub).to(torch.int64), num_classes=4).detach().numpy()
"""
seg_met_sys = np.argmax(out_image_es.detach().numpy(), axis=1)

seg_sys = torch.nn.functional.one_hot(torch.as_tensor(seg_met_sys), num_classes=4).detach().numpy()
ref_sys = torch.nn.functional.one_hot(Tensor(gt_test_es_sub).to(torch.int64), num_classes=4).detach().numpy()
"""

#%% Loss and accuracy
out_mean = res_ed[0] # import mean from model
out_one  = res_ed[1] 

train_loss = out_mean[0]
eval_loss = out_mean[1]

train_loss_0 = out_one[0][0]
eval_loss_0 = out_one[1][0]
train_loss_1 = out_one[0][1]
eval_loss_1 = out_one[1][1]
train_loss_2 = out_one[0][2]
eval_loss_2 = out_one[1][2]
train_loss_3 = out_one[0][3]
eval_loss_3 = out_one[1][3]
train_loss_4 = out_one[0][4]
eval_loss_4 = out_one[1][4]
train_loss_5 = out_one[0][5]
eval_loss_5 = out_one[1][5]

train_acc = out_mean[2]
eval_acc = out_mean[3]

train_acc_1 = out_one[2][0]
eval_acc_1 = out_one[3][0]
train_acc_2 = out_one[2][1]
eval_acc_2 = out_one[3][1]
train_acc_3 = out_one[2][2]
eval_acc_3 = out_one[3][2]
train_acc_4 = out_one[2][3]
eval_acc_4 = out_one[3][3]
train_acc_5 = out_one[2][4]
eval_acc_5 = out_one[3][4]
train_acc_6 = out_one[2][5]
eval_acc_6 = out_one[3][5]

train_inc = (out_mean[4])
eval_inc = (out_mean[5])*5
#%%
t1 = np.arange(len(train_acc_1[0:200]))

plt.figure(figsize=(12, 12),dpi=400)
plt.plot(t1, train_acc_1[0:150], 'b', label = 'Training Loss')
plt.plot(t1 , train_acc_2[0:150], 'g', label = 'Training Loss')
plt.plot(t1 , train_acc_3[0:150], 'r', label = 'Training Loss')
plt.plot(t1 , train_acc_4[0:150], 'y', label = 'Training Loss')
plt.plot(t1 , train_acc_5[0:150], 'm', label = 'Training Loss')
plt.plot(t1 , train_acc_6[0:150], 'c', label = 'Training Loss')

plt.plot(t1, eval_acc_1[0:150], 'b' ,linestyle = 'dashed', label = 'Validation Loss')
plt.plot(t1 , eval_acc_2[0:150], 'g',linestyle = 'dashed', label = 'Validation Loss')
plt.plot(t1 , eval_acc_3[0:150], 'r',linestyle = 'dashed', label = 'Validation Loss')
plt.plot(t1 , eval_acc_4[0:150], 'y',linestyle = 'dashed', label = 'Validation Loss')
plt.plot(t1 , eval_acc_5[0:150], 'm',linestyle = 'dashed', label = 'Validation Loss')
plt.plot(t1 , eval_acc_6[0:150], 'c',linestyle = 'dashed', label = 'Validation Loss')


#%% Plot function
epochs_train = np.arange(len(train_loss))
epochs_eval  = np.arange(len(eval_loss))

plt.figure(figsize=(15, 15),dpi=400)
#plt.rcParams.update({'font.size': 26})
plt.subplot(2,2,1)
plt.plot(t1, train_loss_0[0:200], 'b', label = 'Training loss fold 0')
plt.plot(t1 , train_loss_1[0:200], 'g', label = 'Training loss fold 1')
plt.plot(t1 , train_loss_2[0:200], 'r', label = 'Training loss fold 2')
plt.plot(t1 , train_loss_3[0:200], 'y', label = 'Training loss fold 3')
plt.plot(t1 , train_loss_4[0:200], 'm', label = 'Training loss fold 4')
plt.plot(t1 , train_loss_5[0:200], 'c', label = 'Training loss fold 5')

plt.plot(t1, eval_loss_0[0:200], 'b' ,linestyle = 'dashed', label = 'Validation loss fold 0')
plt.plot(t1 , eval_loss_1[0:200], 'g',linestyle = 'dashed', label = 'Validation loss fold 1')
plt.plot(t1 , eval_loss_2[0:200], 'r',linestyle = 'dashed', label = 'Validation loss fold 2')
plt.plot(t1 , eval_loss_3[0:200], 'y',linestyle = 'dashed', label = 'Validation loss fold 3')
plt.plot(t1 , eval_loss_4[0:200], 'm',linestyle = 'dashed', label = 'Validation loss fold 4')
plt.plot(t1 , eval_loss_5[0:200], 'c',linestyle = 'dashed', label = 'Validation loss fold 5')

plt.grid(color='k', linestyle='-', linewidth=0.2)
plt.xticks(np.arange(0, len(t1) + 2, step = 50), fontsize =14)
plt.yticks(fontsize =14)
plt.xlabel('Epochs', fontsize = 16)
plt.ylabel('Cross-Entropy Loss',  fontsize = 16)
plt.legend(loc="upper right", fontsize = 16)
plt.title('Loss function', fontsize =28)

plt.subplot(2,2,2)
plt.plot(t1, train_acc_1[0:200], 'b', label = 'Training accuracy fold 0')
plt.plot(t1 , train_acc_2[0:200], 'g', label = 'Training accuracy fold 1')
plt.plot(t1 , train_acc_3[0:200], 'r', label = 'Training accuracy fold 2')
plt.plot(t1 , train_acc_4[0:200], 'y', label = 'Training accuracy fold 3')
plt.plot(t1 , train_acc_5[0:200], 'm', label = 'Training accuracy fold 4')
plt.plot(t1 , train_acc_6[0:200], 'c', label = 'Training accuracy fold 5')

plt.plot(t1, eval_acc_1[0:200], 'b' ,linestyle = 'dashed', label = 'Validation accuracy fold 0')
plt.plot(t1 , eval_acc_2[0:200], 'g',linestyle = 'dashed', label = 'Validation accuracy fold 1')
plt.plot(t1 , eval_acc_3[0:200], 'r',linestyle = 'dashed', label = 'Validation accuracy fold 2')
plt.plot(t1 , eval_acc_4[0:200], 'y',linestyle = 'dashed', label = 'Validation accuracy fold 3')
plt.plot(t1 , eval_acc_5[0:200], 'm',linestyle = 'dashed', label = 'Validation accuracy fold 4')
plt.plot(t1 , eval_acc_6[0:200], 'c',linestyle = 'dashed', label = 'Validation accuracy fold 5')

plt.grid(color='k', linestyle='-', linewidth=0.2)
plt.xticks(np.arange(0, len(t1) + 2, step = 50), fontsize =14)
plt.yticks(fontsize =14)
plt.xlabel('Epochs', fontsize = 16)
plt.ylabel('Accuracy %',  fontsize = 16)
plt.legend(loc="lower right", fontsize = 16)
plt.title("Accuracy", fontsize =28)

#%%
plt.subplot(2,2,3)
plt.semilogy(epochs_train + 1 , train_inc, 'b', label = 'Training incorrect')
plt.semilogy(epochs_eval  + 1 , eval_inc,  'r' ,linestyle = 'dashed',label = 'Validation incorrect')
plt.grid(color='k', linestyle='-', linewidth=0.2)
plt.xticks(np.arange(0, len(epochs_train) + 2, step = 50), fontsize =14)
plt.yticks(fontsize =14)
plt.xlabel('Epochs', fontsize = 16)
plt.ylabel('Log #incorrect seg',  fontsize = 16)
plt.legend(loc="upper right", fontsize = 16)
plt.title("Number of Incorrect", fontsize =28)



#%% Plot function
epochs_train = np.arange(len(train_loss))
epochs_eval  = np.arange(len(eval_loss))

plt.figure(figsize=(30, 10),dpi=400)
#plt.rcParams.update({'font.size': 26})
plt.subplot(1,3,1)
plt.plot(epochs_train + 1 , train_loss, 'b', label = 'Training Loss')
plt.plot(epochs_eval  + 1 , eval_loss, 'r', label = 'Validation Loss')
plt.grid(color='k', linestyle='-', linewidth=0.2)
plt.xticks(np.arange(0, len(epochs_train) + 2, step = 50), fontsize =18)
plt.yticks(fontsize =18)
plt.xlabel('Epochs', fontsize = 20)
plt.ylabel('Cross-Entropy Loss',  fontsize = 20)
plt.legend(loc="upper right", fontsize = 20)
plt.title('Loss function', fontsize =28)

plt.subplot(1,3,2)
plt.plot(epochs_train + 1 , train_acc, 'b', label = 'Training accuracy')
plt.plot(epochs_eval  + 1 , eval_acc,  'r',label = 'Validation accuracy')
plt.grid(color='k', linestyle='-', linewidth=0.2)
plt.xticks(np.arange(0, len(epochs_train) + 2, step = 50), fontsize =18)
plt.yticks(fontsize =18)
plt.xlabel('Epochs', fontsize = 20)
plt.ylabel('Accuracy %',  fontsize = 20)
plt.legend(loc="lower right", fontsize = 20)
plt.title("Accuracy", fontsize =28)

plt.subplot(1,3,3)
plt.semilogy(epochs_train + 1 , train_inc, 'b', label = 'Training incorrect')
plt.semilogy(epochs_eval  + 1 , eval_inc,  'r' ,linestyle = 'dashed',label = 'Validation incorrect')
plt.grid(color='k', linestyle='-', linewidth=0.2)
plt.xticks(np.arange(0, len(epochs_train) + 2, step = 50), fontsize =18)
plt.yticks(fontsize =18)
plt.xlabel('Epochs', fontsize = 20)
plt.ylabel('Log #incorrect seg',  fontsize = 20)
plt.legend(loc="upper right", fontsize = 20)
plt.title("Number of Incorrect", fontsize =28)


#%% Plot softmax probabilities for a single slice
test_slice = 12
out_img_ed = np.squeeze(out_image_ed[test_slice,:,:,:].detach().numpy())
alpha = 0.4

fig = plt.figure()

class_title = ['Background','Right Ventricle','Myocardium','Left Ventricle']
plt.figure(dpi=200, figsize=(15,15))
for i in range(0,4):
    plt.suptitle('Diastolic phase: test image at slice %i' %test_slice, fontsize=20)
    plt.subplot(3, 4, i+1)
    plt.subplots_adjust(hspace = 0.05, wspace = 0.2)
    plt.imshow(out_img_ed[i,:,:])
    plt.imshow(im_test_ed_sub[test_slice,0,:,:],alpha=alpha)
    plt.title(class_title[i], fontsize =16)
    plt.xticks(
    rotation=40,
    fontweight='light',
    fontsize=7)
    plt.yticks(
    horizontalalignment='right',
    fontweight='light',
    fontsize=7)
   
    if i == 0:
        plt.ylabel('Softmax probability', fontsize=14)
        
    plt.subplot(3, 4, i+1+4)
    plt.subplots_adjust(hspace = 0.05, wspace = 0.2)
    plt.imshow(seg_dia[test_slice,:,:,i])
    plt.imshow(im_test_ed_sub[test_slice,0,:,:],alpha=alpha)
    if i == 0:
        plt.ylabel('Argmax', fontsize=14)
    plt.subplot(3, 4, i+1+8)     
    plt.subplots_adjust(hspace = 0.05, wspace = 0.2)
    plt.imshow(ref_dia[test_slice,:,:,i])
    plt.imshow(im_test_ed_sub[test_slice,0,:,:],alpha=alpha)
    if i == 0:
        plt.ylabel('Reference', fontsize=14)
plt.show()   

#%% Plot softmax probabilities for a single slice
test_slice = 24
out_img_es = np.squeeze(out_image_es[test_slice,:,:,:].detach().numpy())

fig = plt.figure()
alpha = 0.4
class_title = ['Background','Right Ventricle','Myocardium','Left Ventricle']
plt.figure(dpi=200, figsize=(15,15))
for i in range(0,4):
    plt.suptitle('Systolic phase: test image at slice %i' %test_slice, fontsize=20)
    plt.subplot(3, 4, i+1)
    plt.subplots_adjust(hspace = 0.05, wspace = 0.2)
    plt.imshow(out_img_es[i,:,:])
    plt.imshow(im_test_es_sub[test_slice,0,:,:],alpha=alpha)
    plt.title(class_title[i], fontsize =16)
    plt.xticks(
    rotation=40,
    fontweight='light',
    fontsize=7)
    plt.yticks(
    horizontalalignment='right',
    fontweight='light',
    fontsize=7)
    if i == 0:
        plt.ylabel('Softmax probability', fontsize=14)
        
    plt.subplot(3, 4, i+1+4)
    plt.subplots_adjust(hspace = 0.05, wspace = 0.2)
    plt.imshow(seg_sys[test_slice,:,:,i])
    plt.imshow(im_test_es_sub[test_slice,0,:,:],alpha=alpha)
    if i == 0:
        plt.ylabel('Argmax', fontsize=14)
    plt.subplot(3, 4, i+1+8)
    plt.subplots_adjust(hspace = 0.05, wspace = 0.2)
    plt.imshow(ref_sys[test_slice,:,:,i])
    plt.imshow(im_test_es_sub[test_slice,0,:,:],alpha=alpha)
    if i == 0:
        plt.ylabel('Reference', fontsize=14)
#plt.show()   




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




