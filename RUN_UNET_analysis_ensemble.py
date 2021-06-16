# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 11:41:06 2021

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
import scipy.ndimage


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
    #torchsummary.summary(model, (1, 128, 128))

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
    
    
from load_data_gt_im_sub import load_data_sub

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
#%%
H = 128
W = 128
CV_folds = 6
data_im = im_test_ed_sub.shape[0]


out_soft = np.zeros((CV_folds, data_im, 4, H, W))

im_data = torch.utils.data.DataLoader(im_test_ed_sub, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, prefetch_factor=2, num_workers=0)

for fold in range(0,6):
    if user == 'GPU':
        path_model ='/home/michala/Speciale2021/Speciale2021/Trained_Unet_dice_lv_dia_200_fold{}.pt'.format(fold)
    if user == 'K':
        path_model = 'C:/Users/katrine/Desktop/Optuna/Trained_Unet_CE_sys_200_fold{}.pt'.format(fold)
    model = torch.load(path_model, map_location=torch.device(device))
    model.eval()
    for i, (im) in enumerate(im_data):
        im = Tensor.numpy(im)
        
        if device == 'cuda':
            out = model(Tensor(im).cuda())
        else:
            out = model(Tensor(im))
        out_soft[fold,i,:,:,:] = out["softmax"].detach().cpu().numpy() 
        
    del path_model, model, out
    print('Done for fold',fold)

if user == 'GPU':
    PATH_out_soft = '/home/michala/Speciale2021/Speciale2021/Out_softmax_fold_avg_200dia_dice_lv.pt'
if user == 'K':
    PATH_out_soft = 'C:/Users/katrine/Desktop/Optuna/Out_softmax_fold_avg_200dia_dice_lc.pt'
    
torch.save(out_soft, PATH_out_soft)

""" OUT-COMMENTED PLOT STATEMENTS
#%% Run model0
path_model_0 = 'C:/Users/katrine/Desktop/Optuna/Trained_Unet_CE_dia_fold0.pt'
model_0 = torch.load(path_model_0, map_location=torch.device('cpu'))

model_0.eval()
out_0 = model_0(Tensor(im_test_ed_sub))
out_0 = out_0["softmax"]

#%% Run 
path_model_1 = 'C:/Users/katrine/Desktop/Optuna/Trained_Unet_CE_dia_fold1.pt'
model_1 = torch.load(path_model_1, map_location=torch.device('cpu'))

model_1.eval()
out_1 = model_1(Tensor(im_test_ed_sub))
out_1 = out_1["softmax"].detach().numpy()

#%% Run model2
path_model_2 = 'C:/Users/katrine/Desktop/Optuna/Trained_Unet_CE_dia_fold2.pt'
model_2 = torch.load(path_model_2, map_location=torch.device('cpu'))

model_2.eval()
out_2 = model_2(Tensor(im_test_ed_sub))
out_2 = out_2["softmax"].detach().numpy()

#%% Run model3
path_model_3 = 'C:/Users/katrine/Desktop/Optuna/Trained_Unet_CE_dia_fold3.pt'
model_3 = torch.load(path_model_3, map_location=torch.device('cpu'))

model_3.eval()
out_3 = model_3(Tensor(im_test_ed_sub))
out_3 = out_3["softmax"].detach().numpy()

#%% Run model4
path_model_4 = 'C:/Users/katrine/Desktop/Optuna/Trained_Unet_CE_dia_fold4.pt'
model_4 = torch.load(path_model_4, map_location=torch.device('cpu'))

model_4.eval()
out_4 = model_4(Tensor(im_test_ed_sub))
out_4 = out_4["softmax"].detach().numpy()

#%% Run model5
path_model_5 = 'C:/Users/katrine/Desktop/Optuna/Trained_Unet_CE_dia_fold5.pt'
model_5 = torch.load(path_model_5, map_location=torch.device('cpu'))

model_5.eval()
out_5 = model_5(Tensor(im_test_ed_sub))
out_5 = out_5["softmax"].detach().numpy()
"""
#%% Load model if avergared on GPU
"""
#path_out_soft = '/Users/michalablicher/Desktop/Out_softmax_fold_avg_200dia_dice.pt'
path_out_soft = 'C:/Users/katrine/Desktop/Optuna/Out_softmax_fold_avg_200dia_dice_lc.pt'

out_soft = torch.load(path_out_soft ,  map_location=torch.device(device))

#%%
#Plot softmax probabilities for a single slice
test_slice = 10
alpha = 0.4

fig = plt.figure()

class_title = ['Background','Right Ventricle','Myocardium','Left Ventricle']
ref_dia = torch.nn.functional.one_hot(Tensor(gt_test_ed_sub).to(torch.int64), num_classes=4)
plt.figure(dpi=200, figsize=(18,32))

w = 0.1

for fold_model in range (0,6):
    out_img_ed = np.squeeze(out_soft[fold_model,test_slice,:,:,:])
    seg_met_dia = np.argmax(out_soft[fold_model,:,:,:], axis=1)
    seg_dia = torch.nn.functional.one_hot(torch.as_tensor(seg_met_dia), num_classes=4)
    
    #Reference annotation
    plt.suptitle('Diastolic phase: test image at slice %i for CV folds' %test_slice, fontsize=30, y=0.9)
    plt.subplot(7, 4, 1)
    plt.subplots_adjust(hspace = 0.05, wspace = w)
    plt.imshow(ref_dia[test_slice,:,:,0])
    plt.imshow(im_test_ed_sub[test_slice,0,:,:],alpha=alpha)
    plt.ylabel('Reference', fontsize=16)
    plt.title('Background', fontsize=16)
    
    plt.subplot(7, 4, 2)
    plt.subplots_adjust(hspace = 0.05, wspace = w)
    plt.imshow(ref_dia[test_slice,:,:,1])
    plt.imshow(im_test_ed_sub[test_slice,0,:,:],alpha=alpha)
    plt.title('Right ventricle', fontsize=16)
    
    plt.subplot(7, 4, 3)
    plt.subplots_adjust(hspace = 0.05, wspace = w)
    plt.imshow(ref_dia[test_slice,:,:,2])
    plt.imshow(im_test_ed_sub[test_slice,0,:,:],alpha=alpha)
    plt.title('Myocardium', fontsize=16)
    
    plt.subplot(7, 4, 4)
    plt.subplots_adjust(hspace = 0.05, wspace = w)
    plt.imshow(ref_dia[test_slice,:,:,3])
    plt.imshow(im_test_ed_sub[test_slice,0,:,:],alpha=alpha)
    plt.title('Left ventricle', fontsize=16)
    
    
    #CV model segmentations
    plt.subplot(7, 4, 1+4*(fold_model+1))
    plt.subplots_adjust(hspace = 0.05, wspace = w)
    plt.imshow(seg_dia[test_slice,:,:,0])
    plt.imshow(im_test_ed_sub[test_slice,0,:,:],alpha=alpha)
    plt.ylabel('CV fold {}'.format(fold_model), fontsize=16)
    
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

#%% Mean + argmax + one hot

out_soft_mean = out_soft.mean(axis=0)
out_seg_mean_am = np.argmax(out_soft_mean, axis=1)
out_seg_mean = torch.nn.functional.one_hot(torch.as_tensor(out_seg_mean_am), num_classes=4).detach().cpu().numpy()

ref = torch.nn.functional.one_hot(torch.as_tensor(Tensor(gt_test_ed_sub).to(torch.int64)), num_classes=4).detach().cpu().numpy()

test_slice = 300
plt.imshow(out_seg_mean_am[test_slice,:,:])
#%%
w = 0.1
h = 0.3

plt.figure(dpi=200)
plt.suptitle('Diastolic - Averaged model for test image at slice: {}'.format(test_slice))

plt.subplot(2,2,1)
plt.subplots_adjust(hspace = h, wspace = w)
plt.imshow(out_seg_mean[test_slice,:,:,0])
plt.imshow(im_test_ed_sub[test_slice,0,:,:],alpha=alpha)
plt.title('Background', fontsize=10)

plt.subplot(2,2,2)
plt.subplots_adjust(hspace = h, wspace = w)
plt.imshow(out_seg_mean[test_slice,:,:,1])
plt.imshow(im_test_ed_sub[test_slice,0,:,:],alpha=alpha)
plt.title('Right ventricle', fontsize=10)

plt.subplot(2,2,3)
plt.subplots_adjust(hspace = h, wspace = w)
plt.imshow(out_seg_mean[test_slice,:,:,2])
plt.imshow(im_test_ed_sub[test_slice,0,:,:],alpha=alpha)
plt.title('Myocardium', fontsize=10)

plt.subplot(2,2,4)
plt.subplots_adjust(hspace = h, wspace = w)
plt.imshow(out_seg_mean[test_slice,:,:,3])
plt.imshow(im_test_ed_sub[test_slice,0,:,:],alpha=alpha)
plt.title('Left ventricle', fontsize=10)

#%% Metrics
os.chdir("C:/Users/katrine/Documents/GitHub/Speciale2021")
#os.chdir("/Users/michalablicher/Documents/GitHub/Speciale2021")
from metrics import EF_calculation, dc, hd, jc, precision, mcc, recall, risk, sensitivity, specificity, true_negative_rate, true_positive_rate, positive_predictive_value, hd95, assd, asd, ravd, volume_correlation, volume_change_correlation, obj_assd, obj_asd, obj_fpr, obj_tpr

dice = np.zeros((out_seg_mean.shape[0],3))
haus = np.zeros((out_seg_mean.shape[0],3))

# OBS OBS OBS OBS
# dim[0] = BG
# dim[1] = RV
# dim[2] = MYO
# dim[3] = LV

for i in range(0,out_seg_mean.shape[0]):
      
    dice[i,0] = dc(out_seg_mean[i,:,:,1],ref[i,:,:,1])  # = RV
    dice[i,1] = dc(out_seg_mean[i,:,:,2],ref[i,:,:,2])  # = MYO
    dice[i,2] = dc(out_seg_mean[i,:,:,3],ref[i,:,:,3])  # = LV
    
    # If there is no prediction or annotation then don't calculate Hausdorff distance and
    # skip to calculation for next class
    h_count = 0
    
    if len(np.unique(ref[i,:,:,1]))!=1 and len(np.unique(out_seg_mean[i,:,:,1]))!=1:
        haus[i,0]    = hd(out_seg_mean[i,:,:,1],ref[i,:,:,1])  
        h_count += 1
    else:
        pass
    
    if len(np.unique(ref_dia[i,:,:,2]))!=1 and len(np.unique(out_seg_mean[i,:,:,2]))!=1:      
        haus[i,1]    = hd(out_seg_mean[i,:,:,2],ref[i,:,:,2])  
        h_count += 1
    else:
        pass
    
    if len(np.unique(ref_dia[i,:,:,3]))!=1 and len(np.unique(out_seg_mean[i,:,:,3]))!=1:
        haus[i,2]    = hd(out_seg_mean[i,:,:,3],ref[i,:,:,3])  
        h_count += 1
    else:
        pass
    
        pass        
    if h_count!= 3:
        print('Haus not calculated for all classes for slice: ', i)
    else:
        pass 
    
mean_dice = np.mean(dice, axis=0)  
std_dice = np.std(dice,  axis=0)

mean_haus = np.mean(haus, axis=0)
std_haus = np.std(haus,  axis=0)

print('mean dice = ',mean_dice)  
print('std dice = ', std_dice) 

print('mean haus = ',mean_haus)
print('std haus = ', std_haus)

#%%%%%%%%%%%%%%%%%%%%%%% METRICS %%%%%%%%%%%%%%%%%%%%%
# Slices per patient
p = []

p.append(data_gt_ed_DCM[num_eval_sub:num_test_sub][0].shape[0])
p.append(data_gt_ed_DCM[num_eval_sub:num_test_sub][1].shape[0])
p.append(data_gt_ed_DCM[num_eval_sub:num_test_sub][2].shape[0])
p.append(data_gt_ed_DCM[num_eval_sub:num_test_sub][3].shape[0])
p.append(data_gt_ed_DCM[num_eval_sub:num_test_sub][4].shape[0])
p.append(data_gt_ed_DCM[num_eval_sub:num_test_sub][5].shape[0])
p.append(data_gt_ed_DCM[num_eval_sub:num_test_sub][6].shape[0])
p.append(data_gt_ed_DCM[num_eval_sub:num_test_sub][7].shape[0])


p.append(data_gt_ed_HCM[num_eval_sub:num_test_sub][0].shape[0])
p.append(data_gt_ed_HCM[num_eval_sub:num_test_sub][1].shape[0])
p.append(data_gt_ed_HCM[num_eval_sub:num_test_sub][2].shape[0])
p.append(data_gt_ed_HCM[num_eval_sub:num_test_sub][3].shape[0])
p.append(data_gt_ed_HCM[num_eval_sub:num_test_sub][4].shape[0])
p.append(data_gt_ed_HCM[num_eval_sub:num_test_sub][5].shape[0])
p.append(data_gt_ed_HCM[num_eval_sub:num_test_sub][6].shape[0])
p.append(data_gt_ed_HCM[num_eval_sub:num_test_sub][7].shape[0])

p.append(data_gt_ed_MINF[num_eval_sub:num_test_sub][0].shape[0])
p.append(data_gt_ed_MINF[num_eval_sub:num_test_sub][1].shape[0])
p.append(data_gt_ed_MINF[num_eval_sub:num_test_sub][2].shape[0])
p.append(data_gt_ed_MINF[num_eval_sub:num_test_sub][3].shape[0])
p.append(data_gt_ed_MINF[num_eval_sub:num_test_sub][4].shape[0])
p.append(data_gt_ed_MINF[num_eval_sub:num_test_sub][5].shape[0])
p.append(data_gt_ed_MINF[num_eval_sub:num_test_sub][6].shape[0])
p.append(data_gt_ed_MINF[num_eval_sub:num_test_sub][7].shape[0])


p.append(data_gt_ed_NOR[num_eval_sub:num_test_sub][0].shape[0])
p.append(data_gt_ed_NOR[num_eval_sub:num_test_sub][1].shape[0])
p.append(data_gt_ed_NOR[num_eval_sub:num_test_sub][2].shape[0])
p.append(data_gt_ed_NOR[num_eval_sub:num_test_sub][3].shape[0])
p.append(data_gt_ed_NOR[num_eval_sub:num_test_sub][4].shape[0])
p.append(data_gt_ed_NOR[num_eval_sub:num_test_sub][5].shape[0])
p.append(data_gt_ed_NOR[num_eval_sub:num_test_sub][6].shape[0])
p.append(data_gt_ed_NOR[num_eval_sub:num_test_sub][7].shape[0])

p.append(data_gt_ed_RV[num_eval_sub:num_test_sub][0].shape[0])
p.append(data_gt_ed_RV[num_eval_sub:num_test_sub][1].shape[0]) 
p.append(data_gt_ed_RV[num_eval_sub:num_test_sub][2].shape[0])
p.append(data_gt_ed_RV[num_eval_sub:num_test_sub][3].shape[0]) 
p.append(data_gt_ed_RV[num_eval_sub:num_test_sub][4].shape[0])
p.append(data_gt_ed_RV[num_eval_sub:num_test_sub][5].shape[0]) 
p.append(data_gt_ed_RV[num_eval_sub:num_test_sub][6].shape[0])
p.append(data_gt_ed_RV[num_eval_sub:num_test_sub][7].shape[0]) 


#%% Calculate volume for diastolic phase
#test_index = data_gt_ed[num_eval:num_test]

PATH_soft_dia_fold = 'C:/Users/katrine/Desktop/Optuna/Out_softmax_fold_avg_200dia.pt'
soft_dia_fold = torch.load(PATH_soft_dia_fold, map_location=torch.device(device))

soft_dia_mean = soft_dia_fold.mean(axis=0)
seg_dia_mean  = np.argmax(soft_dia_mean, axis=1)

seg_dia_oh    = torch.nn.functional.one_hot(torch.as_tensor(seg_dia_mean), num_classes=4).detach().cpu().numpy()
ref_dia_oh    = torch.nn.functional.one_hot(torch.as_tensor(Tensor(gt_test_ed_sub).to(torch.int64)), num_classes=4).detach().cpu().numpy()

test_index = len(p)

s = 0
target_vol_ed = np.zeros(test_index)
ref_vol_ed    = np.zeros(test_index)

for i in range(0,test_index):
    #print('patient nr.', i)
    for j in range(0, p[i]):
        #print('slice # ',j)
        target_vol_ed[i] += np.sum(seg_dia_oh[j+s,:,:,3])
        ref_vol_ed[i]    += np.sum(ref_dia_oh[j+s,:,:,3])
        #print('j+s = ',j+s)
        
    s += p[i] 
    #print('s= ',s)
   
#%% Calculate volume for systolic phase
PATH_soft_sys_fold = 'C:/Users/katrine/Desktop/Optuna/Out_softmax_fold_avg_200sys.pt'
soft_sys_fold = torch.load(PATH_soft_sys_fold, map_location=torch.device(device))

soft_sys_mean = soft_sys_fold.mean(axis=0)
seg_sys_mean  = np.argmax(soft_sys_mean, axis=1)

seg_sys_oh    = torch.nn.functional.one_hot(torch.as_tensor(seg_sys_mean), num_classes=4).detach().cpu().numpy()
ref_sys_oh    = torch.nn.functional.one_hot(torch.as_tensor(Tensor(gt_test_es_sub).to(torch.int64)), num_classes=4).detach().cpu().numpy()

test_index = len(p)

s = 0
target_vol_es = np.zeros(test_index)
ref_vol_es    = np.zeros(test_index)

for i in range(0,test_index):
    #print('patient nr.', i)
    for j in range(0, p[i]):
        #print('slice # ',j)
        target_vol_es[i] += np.sum(seg_sys_oh[j+s,:,:,3])
        ref_vol_es[i]    += np.sum(ref_sys_oh[j+s,:,:,3])
        #print('j+s = ',j+s)
        
    s += p[i] 
    #print('s= ',s)
     
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

print('EF ref  = ', ef_ref[0]) 
print('esv ref = ', ef_ref[1]) 
print('edv ref = ', ef_ref[2]) 

print('EF seg = ', ef_target[0]) 
print('esv seg = ', ef_target[1]) 
print('edv seg = ', ef_target[2]) 

#%%
slice = 169
c= 3

plt.imshow(seg_dia_oh[slice,:,:,c])
plt.imshow(seg_sys_oh[slice,:,:,c], alpha =0.5)

b = np.sum(ref_dia_oh[slice,:,:,c])

#%%
pier = np.corrcoef(ef_ref[2],ef_target[2])

"""
