p#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 11:04:54 2021

@author: michalablicher
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 15:28:28 2021

@author: michalablicher
"""
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
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

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
    unet.cuda()
    #torchsummary.summary(model, (1, 128, 128))
    
#%% Specify directory
cwd = os.getcwd()
#os.chdir("C:/Users/katrine/Documents/Universitet/Speciale/ACDC_training_data/training")   # Local directory katrine
#os.chdir('/Users/michalablicher/Desktop/training')     # Local directory michala
os.chdir("/home/michala/training")                      # Server directory michala


#%% Specify directory
#os.chdir("C:/Users/katrine/Documents/GitHub/Speciale2021")
#os.chdir('/Users/michalablicher/Documents/GitHub/Speciale2021')

from load_data_gt_im_sub import load_data_sub

data_im_es_DCM,  data_gt_es_DCM  = load_data_sub('GPU','Systole','DCM')
data_im_es_HCM,  data_gt_es_HCM  = load_data_sub('GPU','Systole','HCM')
data_im_es_MINF, data_gt_es_MINF = load_data_sub('GPU','Systole','MINF')
data_im_es_NOR,  data_gt_es_NOR  = load_data_sub('GPU','Systole','NOR')
data_im_es_RV,   data_gt_es_RV   = load_data_sub('GPU','Systole','RV')

data_im_ed_DCM,  data_gt_ed_DCM  = load_data_sub('GPU','Diastole','DCM')
data_im_ed_HCM,  data_gt_ed_HCM  = load_data_sub('GPU','Diastole','HCM')
data_im_ed_MINF, data_gt_ed_MINF = load_data_sub('GPU','Diastole','MINF')
data_im_ed_NOR,  data_gt_ed_NOR  = load_data_sub('GPU','Diastole','NOR')
data_im_ed_RV,   data_gt_ed_RV   = load_data_sub('GPU','Diastole','RV')


#%% BATCH GENERATOR
num_train_sub = 16 
num_eval_sub = num_train_sub + 2
num_test_sub = num_eval_sub + 2

im_train_sub = np.concatenate((np.concatenate(data_im_ed_DCM[0:num_train_sub]).astype(None),
                                  np.concatenate(data_im_ed_HCM[0:num_train_sub]).astype(None),
                                  np.concatenate(data_im_ed_MINF[0:num_train_sub]).astype(None),
                                  np.concatenate(data_im_ed_NOR[0:num_train_sub]).astype(None),
                                  np.concatenate(data_im_ed_RV[0:num_train_sub]).astype(None)))

gt_train_sub = np.concatenate((np.concatenate(data_gt_ed_DCM[0:num_train_sub]).astype(None),
                                  np.concatenate(data_gt_ed_HCM[0:num_train_sub]).astype(None),
                                  np.concatenate(data_gt_ed_MINF[0:num_train_sub]).astype(None),
                                  np.concatenate(data_gt_ed_NOR[0:num_train_sub]).astype(None),
                                  np.concatenate(data_gt_ed_RV[0:num_train_sub]).astype(None)))

im_eval_sub = np.concatenate((np.concatenate(data_im_ed_DCM[num_train_sub:num_eval_sub]).astype(None),
                                  np.concatenate(data_im_ed_HCM[num_train_sub:num_eval_sub]).astype(None),
                                  np.concatenate(data_im_ed_MINF[num_train_sub:num_eval_sub]).astype(None),
                                  np.concatenate(data_im_ed_NOR[num_train_sub:num_eval_sub]).astype(None),
                                  np.concatenate(data_im_ed_RV[num_train_sub:num_eval_sub]).astype(None)))

gt_eval_sub = np.concatenate((np.concatenate(data_gt_ed_DCM[num_train_sub:num_eval_sub]).astype(None),
                                  np.concatenate(data_gt_ed_HCM[num_train_sub:num_eval_sub]).astype(None),
                                  np.concatenate(data_gt_ed_MINF[num_train_sub:num_eval_sub]).astype(None),
                                  np.concatenate(data_gt_ed_NOR[num_train_sub:num_eval_sub]).astype(None),
                                  np.concatenate(data_gt_ed_RV[num_train_sub:num_eval_sub]).astype(None)))


im_test_sub = np.concatenate((np.concatenate(data_im_ed_DCM[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_im_ed_HCM[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_im_ed_MINF[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_im_ed_NOR[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_im_ed_RV[num_eval_sub:num_test_sub]).astype(None)))

gt_test_sub = np.concatenate((np.concatenate(data_gt_ed_DCM[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_gt_ed_HCM[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_gt_ed_MINF[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_gt_ed_NOR[num_eval_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_gt_ed_RV[num_eval_sub:num_test_sub]).astype(None)))


#%%

data_train = Tensor((np.squeeze(im_train_sub), gt_train_sub))
data_train_n = data_train.permute(1,0,2,3)

data_eval = Tensor((np.squeeze(im_eval_sub), gt_eval_sub))
data_eval_n = data_eval.permute(1,0,2,3)

batch_size = 40
train_dataloader = DataLoader(data_train_n, batch_size=batch_size, shuffle=True, drop_last=True)
eval_dataloader = DataLoader(data_eval_n, batch_size=batch_size, shuffle=True, drop_last=True)

#im_train , lab_train = next(iter(train_dataloader))
#im_eval , lab_eval   = next(iter(eval_dataloader))


print("The shape of the data loader", len(train_dataloader),
      " should equal to number of images // batch_size:", len(data_train_n),"//", batch_size, "=",len(data_train_n) // batch_size)


print("The shape of the data loader", len(eval_dataloader),
      " should equal to number of images // batch_size:",len(data_eval_n), "//", batch_size, "=",len(data_eval_n) // batch_size )


#%% Setting up training loop
# OBS DECREASED LEARNING RATE AND EPSILON ADDED TO OPTIMIZER

LEARNING_RATE = 0.0001 # 
criterion    = nn.CrossEntropyLoss()
#criterion     = nn.BCELoss()
#criterion     = SoftDice
#criterion     = brier_score_loss()

# weight_decay is equal to L2 regularizationst
optimizer = optim.Adam(unet.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
# torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

# and a learning rate scheduler which decreases the learning rate by 10x every 3 epochs
#lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
#                                               step_size=3,
#                                               gamma=0.1)

num_epoch = 100

#%% Training
train_losses = []
eval_losses  = []
eval_loss    = 0.0
train_loss   = 0.0 #[]

for epoch in range(num_epoch):  # loop over the dataset multiple times
    
    unet.train()
    print('Epoch train =',epoch)
    #0.0  
    for i, (train_data) in enumerate(train_dataloader):
        # get the inputs
        #inputs, labels = data
        inputs = Tensor(np.expand_dims(train_data[:,0,:,:], axis = 1))
        inputs = inputs.cuda()
        labels = train_data[:,1,:,:]
        labels = labels.cuda()
        #print('i=',i)
        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)
        labels = labels.long()
        
        # Clear the gradients
        optimizer.zero_grad()
       
        # Forward Pass
        output = unet(inputs)     
        output = output["log_softmax"]
        #print('output shape = ', output.shape)
        
        # Find loss
        loss = criterion(output, labels)
        #print('loss = ', loss)
        
        # Calculate gradients
        loss.backward()
        
        # Update Weights
        optimizer.step()

        # Calculate loss
        train_loss += loss.item() #.detach().cpu().numpy()
        
    train_losses.append(train_loss/train_data.shape[0]) # This is normalised by batch size
    #train_losses.append(np.mean(batch_loss))
    train_loss = 0.0 #[]
    
    unet.eval()
    print('Epoch eval=',epoch)
     
    for i, (eval_data) in enumerate(eval_dataloader):
        # get the inputs
        #inputs, labels = data
        inputs = Tensor(np.expand_dims(eval_data[:,0,:,:], axis = 1))
        inputs = inputs.cuda()
        labels = eval_data[:,1,:,:]
        labels = labels.cuda()
        #print('i=',i)

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)
        labels = labels.long()
        
        # Forward pass
        output = unet(inputs)     
        output = output["log_softmax"]
        # Find loss
        loss = criterion(output, labels)
        
        # Calculate loss
        #eval_loss.append(loss.item())
        eval_loss += loss.item() #.detach().cpu().numpy()
        
    eval_losses.append(eval_loss/eval_data.shape[0]) # This is normalised by batch size
    #eval_losses.append(np.mean(eval_loss))
    eval_loss = 0.0

print('Finished Training + Evaluation')
        

#%% Plot loss curves

epochs = np.arange(len(train_losses))
epochs_eval = np.arange(len(eval_losses))
plt.figure(dpi=200)
plt.plot(epochs + 1 , train_losses, 'b', label='Training Loss')
plt.plot(epochs_eval + 1 , eval_losses, 'r', label='Validation Loss')
plt.xticks(np.arange(1,num_epoch+1, step = 10))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc="upper right")
plt.title("Loss function")
plt.savefig('/home/michala/Speciale2021/Speciale2021/Trained_Unet_CE_dia_sub_loss.png')
#plt.savefig('/home/katrine/Speciale2021/Speciale2021/Trained_Unet_CE_dia_loss.png')


#%% Plot accuracy curve


#%% Save model
PATH_model = "/home/michala/Speciale2021/Speciale2021/Trained_Unet_CE_dia_sub_batch_100.pt"
PATH_state = "/home/michala/Speciale2021/Speciale2021/Trained_Unet_CE_batch_state.pt"

#PATH_model = "/home/katrine/Speciale2021/Speciale2021/Trained_Unet_CE_dia.pt"
#PATH_state = "/home/katrine/Speciale2021/Speciale2021/Trained_Unet_CE_dia_state.pt"

torch.save(unet, PATH_model)
torch.save(unet.state_dict(), PATH_state)













