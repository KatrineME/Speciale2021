# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 15:05:04 2021

@author: katrine
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% Load packages
import torch
import os
import nibabel as nib
import numpy   as np
import torchvision
import glob2
import torch.optim as optim

from torch.autograd  import Variable
from torch import nn
from torch import Tensor

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
    model = BayesUNet(num_classes=4, in_channels=1, drop_prob=0.1)
    model.cuda()
    #torchsummary.summary(model, (1, 128, 128))
    
#%% Specify directory
cwd = os.getcwd()
#os.chdir("C:/Users/katrine/Documents/Universitet/Speciale/ACDC_training_data/training")   # Local directory katrine
#os.chdir('/Users/michalablicher/Desktop/training')     # Local directory michala
os.chdir("/home/michala/training")                      # Server directory michala

#%% Load image  
frame_im = np.sort(glob2.glob('patient*/**/patient*_frame*[0-9].nii.gz'))
dia      = np.linspace(0,len(frame_im)-2,100).astype(int)

frame_dia_im = frame_im[dia]

num_patients = len(frame_dia_im)
H    = 128
W    = 128
in_c = 1

data_im = []
centercrop = torchvision.transforms.CenterCrop((H,W))

for i in range(0,num_patients):
    nimg = nib.load(frame_dia_im[i])
    img  = nimg.get_fdata()
    
    im_slices      = img.shape[2]
    centercrop_img = Tensor(np.zeros((H,W,im_slices)))
    
    for j in range(0,im_slices):
        centercrop_img[:,:,j] = centercrop(Tensor(img[:,:,j]))
   
    in_image = np.expand_dims(centercrop_img,0)
    in_image = Tensor(in_image).permute(3,0,1,2).detach().numpy()
    
    data_im.append(in_image.astype(object))

#%% Load annotations

frame_gt = np.sort(glob2.glob('patient*/**/patient*_frame*[0-9]_gt.nii.gz'))
frame_dia_gt = frame_gt[dia]

num_patients = len(frame_dia_gt)
H = 128
W = 128

data_gt = [] 
centercrop     = torchvision.transforms.CenterCrop((H,W))

for i in range(0,num_patients):
    n_gt = nib.load(frame_dia_gt[i])
    gt  = n_gt.get_fdata()
    
    gt_slices      = gt.shape[2]
    centercrop_gt = Tensor(np.zeros((H,W,gt_slices)))
    
    for j in range(0,gt_slices):
        centercrop_gt[:,:,j] = centercrop(Tensor(gt[:,:,j]))
   
    in_gt = Tensor(centercrop_gt).permute(2,0,1).detach().numpy()
    
    data_gt.append(in_gt.astype(object))

#%% BATCH GENERATOR
num = 5

num_train = num
num_eval  = num + num_train 
num_test  = num + num_eval

print('length of data_im',len(data_im))

im_flat_train = np.concatenate(data_im[0:num_train]).astype(None)
gt_flat_train = np.concatenate(data_gt[0:num_train]).astype(None)


im_flat_eval = np.concatenate(data_im[num_train:num_eval]).astype(None)
gt_flat_eval = np.concatenate(data_gt[num_train:num_eval]).astype(None)

im_flat_test = np.concatenate(data_im[num_eval:num_test]).astype(None)
gt_flat_test = np.concatenate(data_gt[num_eval:num_test]).astype(None)

#%% Setting up training loop
# OBS DECREASED LEARNING RATE AND EPSILON ADDED TO OPTIMIZER

LEARNING_RATE = 0.0001 # 
criterion    = nn.CrossEntropyLoss() 
#criterion     = nn.BCELoss()
#criterion     = SoftDice
#criterion     = brier_score_loss()

# weight_decay is equal to L2 regularizationst
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-04, weight_decay=1e-4)
# torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

# and a learning rate scheduler which decreases the learning rate by 10x every 3 epochs
#lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
#                                               step_size=3,
#                                               gamma=0.1)

num_epoch = 10

#%% Training
losses = []
losses_eval = []
trainloader = im_flat_train

for epoch in range(num_epoch):  # loop over the dataset multiple times
    
    model.train()
    print('Epoch train =',epoch)
    train_loss = 0.0  
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        #inputs, labels = data
        inputs = Tensor(im_flat_train)
        labels = Tensor(gt_flat_train)
        #print('i=',i)

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)
        labels = labels.long()

        # Clear the gradients
        optimizer.zero_grad()

        # Forward Pass
        output = model(inputs)     
        output = output["log_softmax"]
        
        # Find loss
        loss = criterion(output, labels)
        # Calculate gradients
        loss.backward()
        # Update Weights
        optimizer.step()
        
        # Calculate loss
        train_loss += loss.item()
    losses.append(train_loss/trainloader.shape[0]) # This should be normalised by batch size
    train_loss = 0.0
     
    model.eval()
    print('Epoch eval=',epoch)
    eval_loss = 0.0  
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        #inputs, labels = data
        inputs = Tensor(im_flat_eval)
        labels = Tensor(gt_flat_eval)
        #print('i=',i)

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)
        labels = labels.long()
        
        # Forward pass
        output = model(inputs)     
        output = output["log_softmax"]
        # Find loss
        loss = criterion(output, labels)

        # Calculate loss
        eval_loss += loss.item()
    losses_eval.append(eval_loss/trainloader.shape[0])
        #(epoch + 1, i + 1, eval_loss)
    eval_loss = 0.0

print('Finished Training + Evaluation')
        

#%% Save model
#os.chdir("/home/michala/Speciale2021/Speciale2021/data/training")

PATH_model = "/home/michala/Speciale2021/Speciale2021/trained_Unet_gpu_test.pt"
PATH_state = "/home/michala/Speciale2021/Speciale2021/trained_Unet_gpu_test_state.pt"
torch.save(model, PATH_model)
torch.save(model.state_dict(), PATH_state)
