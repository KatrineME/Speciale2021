# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 11:36:46 2021

@author: katrine
"""
#%% Load packages
import torch
import os
import scipy.stats
import numpy   as np
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor

# BayesUNet
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
os.chdir("C:/Users/katrine/Documents/GitHub/Speciale2021")

from load_data_gt_im_sub import load_data_sub

user = 'K'

data_im_ed_DCM,  data_gt_ed_DCM  = load_data_sub(user,'Diastole','DCM')
data_im_ed_HCM,  data_gt_ed_HCM  = load_data_sub(user,'Diastole','HCM')
data_im_ed_MINF, data_gt_ed_MINF = load_data_sub(user,'Diastole','MINF')
data_im_ed_NOR,  data_gt_ed_NOR  = load_data_sub(user,'Diastole','NOR')
data_im_ed_RV,   data_gt_ed_RV   = load_data_sub(user,'Diastole','RV')



#%% BATCH GENERATOR
num_train_sub = 12
num_eval_sub = num_train_sub
num_test_sub = num_eval_sub + 6

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

#%% Load trained U-Net model
#PATH_model_ed = '/Users/michalablicher/Desktop/Trained_Unet_CE_dia_CrossVal.pt'
PATH_model_ed = 'C:/Users/katrine/Documents/Universitet/Speciale/Trained_Unet_CE_dia_CrossVal_mc01.pt'
unet_ed       = torch.load(PATH_model_ed, map_location=torch.device('cpu'))
#%% Load results and losses from CV loop
PATH_res_ed = '/Users/michalablicher/Desktop/Trained_Unet_CE_dia_train_results.pt'
res_ed = torch.load(PATH_res_ed, map_location=torch.device('cpu'))

#%% Run test on Unet DIASTOLIC
unet_ed.eval()
out_trained_ed = unet_ed(Tensor(im_test_ed_sub))
out_image_ed   = out_trained_ed["softmax"]

#%% Argmax and onehot encoding
seg_met_dia = np.argmax(out_image_ed.detach().numpy(), axis=1)

seg_dia = torch.nn.functional.one_hot(torch.as_tensor(seg_met_dia), num_classes=4).detach().numpy()
ref_dia = torch.nn.functional.one_hot(Tensor(gt_test_ed_sub).to(torch.int64), num_classes=4).detach().numpy()

#%% Plot softmax probabilities for a single slice
test_slice = 9
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

#%% UNCERTAINTY B-MAPS
T = 10 # OBS Set up to find optimal
C = out_image_ed.shape[1]
in_image = Tensor(im_test_ed_sub[0:16,:,:,:])

b = np.zeros((T,in_image.shape[0],C,in_image.shape[2],in_image.shape[3]))

unet_ed.train()
plt.figure(dpi=200)

image = 9
#
for i in range(0,T):
    out_trained_bmap = unet_ed(in_image)
    out_bmap = out_trained_bmap["softmax"]
    b[i,:,:,:,:] = out_bmap.detach().numpy()
    out_img_bmap = np.squeeze(out_bmap[image,:,:,:].detach().numpy())
    
    plt.suptitle('Softmax prob. for each of T iterations at slice 4', fontsize=16)
    plt.subplot(4,T,i+1)
    plt.subplots_adjust(hspace = 0.0, wspace = 0.1)
    plt.xticks(fontsize=7, fontweight='light')
    plt.yticks(fontsize=7, fontweight='light')
    plt.imshow(out_img_bmap[0,:,:])
    plt.axis('off')
    
    plt.subplot(4,T,i+1+T)
    plt.subplots_adjust(hspace = 0.0, wspace = 0.1)
    plt.xticks(fontsize=7, fontweight='light')
    plt.yticks(fontsize=7, fontweight='light')
    plt.imshow(out_img_bmap[1,:,:])
    plt.axis('off')
    
    plt.subplot(4,T,i+1+T*2)
    plt.subplots_adjust(hspace = 0.0, wspace = 0.1)
    plt.xticks(fontsize=7, fontweight='light')
    plt.yticks(fontsize=7, fontweight='light')
    plt.imshow(out_img_bmap[2,:,:])
    plt.axis('off')
    
    plt.subplot(4,T,i+1+T*3)
    plt.subplots_adjust(hspace = 0.0, wspace = 0.1)
    plt.xticks(fontsize=7, fontweight='light')
    plt.yticks(fontsize=7, fontweight='light')
    plt.imshow(out_img_bmap[3,:,:])
    plt.axis('off')

#%%
pred_mean = (1/T)*sum(b,0)
d = (1/(T-1)) * sum((b-pred_mean)**2,0)

b_map      = (1/C) * np.sum(np.sqrt(d),axis=1)
b_map_norm = np.zeros((b_map.shape))
for i in range(0,b_map.shape[0]):
    b_map_norm[i,:,:] = b_map[i,:,:]/np.max(b_map,axis=(1,2))[i]
#%% Plot B-maps
#plt.figure(figsize=(10,10))
subp = b_map.shape[0]

plt.figure(dpi=200)
for i in range(0, subp):
    plt.suptitle('B-maps', fontsize =16)
    fig_test = plt.subplot(np.ceil(np.sqrt(subp)),np.ceil(np.sqrt(subp)),i+1)
    plt.subplots_adjust(hspace = 0.50, wspace = 0.35)
    plt.imshow(b_map_norm[i,:,:]) 
    plt.xticks(fontsize=0,fontweight='light')
    plt.yticks(fontsize=0,fontweight='light')
    plt.title('Slice %i' %i, fontsize =7) 
    plt.clim(0, np.max(b_map_norm))
    cbar = plt.colorbar(fraction=0.045)
    cbar.ax.tick_params(labelsize=5)
    cbar.ax.locator_params(nbins=5)
    #cbar.ax.set_yticklabels(['low','','high'])
    fig.tight_layout()
    plt.show
       
#%% Uncertainty maps (E-maps)   
emap = np.zeros((in_image.shape[0],in_image.shape[2],in_image.shape[3]))
subp = emap.shape[0]

plt.figure(dpi=200)
for i in range(0, subp):
    plt.suptitle('E-maps', fontsize=16)
    plt.subplot(np.ceil(np.sqrt(subp)),np.ceil(np.sqrt(subp)),i+1)
    plt.subplots_adjust(hspace = 0.50, wspace = 0.4)
    out_img = (out_image_ed [i,:,:].detach().numpy())
    entropy2 = scipy.stats.entropy(out_img)
    
    # Normalize 
    m_entropy   = np.max(entropy2)
    print(m_entropy)
    entropy     = entropy2/m_entropy
    emap[i,:,:] = entropy

    plt.imshow(entropy) 
    plt.xticks(fontsize=0, fontweight='light')
    plt.yticks(fontsize=0, fontweight='light')
    plt.title('Slice %i' %i, fontsize =7) 
    cbar = plt.colorbar(fraction=0.045)
    cbar.ax.tick_params(labelsize=5) 
    cbar.ax.locator_params(nbins=5)
    fig.tight_layout()
    plt.show
# %% #################################################################################################################################
#    Thresholding of uncertainties
# %% #################################################################################################################################














