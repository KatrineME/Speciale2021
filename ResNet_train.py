# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 13:24:53 2021

@author: katrine
"""
import torch
import torch.nn as nn
import math
import os
import numpy as np
import torch.optim as optim
from torch.autograd  import Variable
import matplotlib.pyplot as plt
from torch import Tensor
import torch.utils.model_zoo as model_zoo
BatchNorm = nn.BatchNorm2d
DropOut = nn.Dropout2d



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.zero_()


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride,
                             padding=dilation[0], dilation=dilation[0])
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes,
                             padding=dilation[1], dilation=dilation[1])
        self.bn2 = BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)

        return out


class DRN(nn.Module):

    def __init__(self, block, layers, num_classes=1000,
                 input_channels=3,
                 channels=(16, 32, 64, 128, 256, 512, 512, 512),
                 out_map=False, out_middle=False, pool_size=28, arch='D', mc_dropout=False):
        super(DRN, self).__init__()
        self.inplanes = channels[0]
        self.out_map = out_map
        self.out_dim = channels[-1]
        self.out_middle = out_middle
        self.arch = arch
        self.mc_dropout = mc_dropout
        # print("DRN - Parameter info {} {} {} {} {}".format(out_map, self.out_dim, out_middle, arch, mc_dropout))

        if arch == 'C':
            self.conv1 = nn.Conv2d(input_channels, channels[0], kernel_size=7, stride=1,
                                   padding=3, bias=False)
            self.bn1 = BatchNorm(channels[0])
            self.relu = nn.ReLU(inplace=True)

            self.layer1 = self._make_layer(
                BasicBlock, channels[0], layers[0], stride=1)
            self.layer2 = self._make_layer(
                BasicBlock, channels[1], layers[1], stride=2)
        elif arch == 'D':
            self.layer0 = nn.Sequential(
                nn.Conv2d(input_channels, channels[0], kernel_size=7, stride=1, padding=3,
                          bias=False),
                BatchNorm(channels[0]),
                nn.ReLU(inplace=True)
            )

            self.layer1 = self._make_conv_layers(
                channels[0], layers[0], stride=1)
            # 16-07 jorg
            if self.mc_dropout:
                self.layer1.add_module("dropout_layer", DropOut(p=0.1))
            self.layer2 = self._make_conv_layers(
                channels[1], layers[1], stride=2)
            if self.mc_dropout:
                self.layer2.add_module("dropout_layer", DropOut(p=0.1))

        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2)
        if self.mc_dropout:
            self.layer3.add_module("dropout_layer", DropOut(p=0.1))
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2)
        if self.mc_dropout:
            self.layer4.add_module("dropout_layer", DropOut(p=0.1))
        self.layer5 = self._make_layer(block, channels[4], layers[4],
                                       dilation=2, new_level=False)
        if self.mc_dropout:
            self.layer5.add_module("dropout_layer", DropOut(p=0.1))
        self.layer6 = None if layers[5] == 0 else \
            self._make_layer(block, channels[5], layers[5], dilation=4,
                             new_level=False)
        if self.mc_dropout:
            self.layer6.add_module("dropout_layer", DropOut(p=0.1))

        if arch == 'C':
            self.layer7 = None if layers[6] == 0 else \
                self._make_layer(BasicBlock, channels[6], layers[6], dilation=2,
                                 new_level=False, residual=False)
            self.layer8 = None if layers[7] == 0 else \
                self._make_layer(BasicBlock, channels[7], layers[7], dilation=1,
                                 new_level=False, residual=False)
        elif arch == 'D':
            self.layer7 = None if layers[6] == 0 else \
                self._make_conv_layers(channels[6], layers[6], dilation=2)
            if self.mc_dropout:
                self.layer7.add_module("dropout_layer", DropOut(p=0.1))
            self.layer8 = None if layers[7] == 0 else \
                self._make_conv_layers(channels[7], layers[7], dilation=1)
            if self.mc_dropout:
                self.layer8.add_module("dropout_layer", DropOut(p=0.1))

        if num_classes > 0:
            # Note: this is only used in case self.out_map is True
            self.avgpool = nn.AvgPool2d(pool_size)
            self.fc = nn.Conv2d(self.out_dim, num_classes, kernel_size=1,
                                stride=1, padding=0, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
                    new_level=True, residual=True):
        assert dilation == 1 or dilation % 2 == 0
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = list()
        layers.append(block(
            self.inplanes, planes, stride, downsample,
            dilation=(1, 1) if dilation == 1 else (
                dilation // 2 if new_level else dilation, dilation),
            residual=residual))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, residual=residual,
                                dilation=(dilation, dilation)))

        return nn.Sequential(*layers)

    def _make_conv_layers(self, channels, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(self.inplanes, channels, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                BatchNorm(channels),
                nn.ReLU(inplace=True)])
            self.inplanes = channels
        return nn.Sequential(*modules)

    def forward(self, x):
        y = list()

        if self.arch == 'C':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        elif self.arch == 'D':
            x = self.layer0(x)

        x = self.layer1(x)
        y.append(x)
        x = self.layer2(x)
        y.append(x)

        x = self.layer3(x)
        y.append(x)

        x = self.layer4(x)
        y.append(x)

        x = self.layer5(x)
        y.append(x)

        if self.layer6 is not None:
            x = self.layer6(x)
            y.append(x)

        if self.layer7 is not None:
            x = self.layer7(x)
            y.append(x)

        if self.layer8 is not None:
            x = self.layer8(x)
            y.append(x)

        if self.out_map:
            x = self.fc(x)
        else:
            x = self.avgpool(x)
            x = self.fc(x)
            x = x.view(x.size(0), -1)

        if self.out_middle:
            return x, y
        else:
            return x



class SimpleRSN(nn.Module):

    def __init__(self, block, channels=(16, 32, 64, 128), n_channels_input=3, n_classes=2, drop_prob=0.):
        super(SimpleRSN, self).__init__()
        self.inplanes = channels[0]
        self.out_dim = channels[-1]
        self.drop_prob = drop_prob
        self.use_dropout = True if drop_prob > 0. else False

        self.layer0 = nn.Sequential(
            nn.Conv2d(n_channels_input, channels[0], kernel_size=7, stride=1, padding=3,
                      bias=False),
            BatchNorm(channels[0]),
            nn.ReLU(inplace=True)
        )

        self.layer1 = self._make_conv_layers(channels[1], stride=1)
        self.layer2 = self._make_layer(block, channels[1], stride=2)
        self.layer3 = self._make_layer(block, channels[2], stride=2)
        self.layer4 = self._make_layer(block, channels[3], stride=2)
        self.classifier = nn.Sequential(
            nn.Conv2d(self.inplanes, self.out_dim, kernel_size=1, padding=0),
            nn.ReLU(True),
            nn.Dropout(p=self.drop_prob),
            nn.Conv2d(self.out_dim, self.out_dim, kernel_size=1, padding=0),
            nn.ReLU(True),
            nn.Dropout(p=self.drop_prob),
            nn.Conv2d(self.out_dim, n_classes, kernel_size=1, padding=0),
        )
        self.softmax_layer = nn.Softmax(dim=1)
        self.log_softmax_layer = nn.LogSoftmax(dim=1)
        self.classifier.apply(weights_init)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        y = self.classifier(x)
        return {'log_softmax': self.log_softmax_layer(y), 'softmax': self.softmax_layer(y)}

    def _make_conv_layers(self, channels, stride=1, dilation=1):
        modules = []
        modules.extend([
                nn.Conv2d(self.inplanes, channels, kernel_size=3,
                          stride=stride,
                          padding=dilation, bias=False, dilation=dilation),
                BatchNorm(channels),
                nn.ReLU(inplace=True),
                nn.Dropout(p=self.drop_prob)])
        self.inplanes = channels
        return nn.Sequential(*modules)

    def _make_layer(self, block, planes, stride=1, dilation=1,
                    new_level=True, residual=True):
        assert dilation == 1 or dilation % 2 == 0
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = list()
        layers.append(block(
            self.inplanes, planes, stride, downsample,
            dilation=(1, 1) if dilation == 1 else (
                dilation // 2 if new_level else dilation, dilation),
            residual=residual))
        if self.use_dropout:
            layers.append(nn.Dropout(p=self.drop_prob))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)


class CombinedRSN(SimpleRSN):

    def __init__(self, BasicBlock, channels=(16, 32, 64, 128), n_channels_input=3, n_classes=2, drop_prob=0.3):
        super(CombinedRSN, self).__init__(block=BasicBlock, channels=channels, n_channels_input=n_channels_input,
                                          n_classes=n_classes, drop_prob=drop_prob)

        self.up = lambda inp, shape: nn.functional.interpolate(inp, size=shape, mode='bilinear', align_corners=False)
        self.seg = nn.Conv2d(self.out_dim, n_classes,
                             kernel_size=1, bias=True)
        self.seg.apply(weights_init)

    def forward(self, x):
        output_size = x.shape[2:]
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        features = self.layer4(x)
        x = self.classifier(features)
        y = self.seg(features)
        y = self.up(y, output_size)

        return {'log_softmax': self.log_softmax_layer(x), 'softmax': self.softmax_layer(x),
                'log_softmax_y': self.log_softmax_layer(y), 'softmax_y': self.softmax_layer(y).detach().cpu().numpy()}


if __name__ == "__main__":
    import torchsummary

    n_channels = 3  # 3
    n_classes  = 2
    model  = CombinedRSN(BasicBlock, channels=(16, 32, 64, 128), n_channels_input=n_channels, n_classes=n_classes, drop_prob=0.3)
    #model = SimpleRSN(BasicBlock, channels=(16, 32, 64, 128), n_channels_input=n_channels, n_classes=n_classes, drop_prob=0.5)
    #model.cuda()
    torchsummary.summary(model, (n_channels, 80, 80))
    
#%% Specify directory
os.chdir("C:/Users/katrine/Documents/GitHub/Speciale2021")
#os.chdir('/Users/michalablicher/Documents/GitHub/Speciale2021')

from load_data_gt_im import load_data

data_im_es, data_gt_es = load_data('K','Systole')
data_im_ed, data_gt_ed = load_data('K','Diastole')

#%% Test normal patients

nor = 60
num_train = nor + 5#0
num_eval  = 3#0
num_test  = 10#0

lim_eval  = num_train + num_eval
lim_test  = lim_eval + num_test

im_flat_test_es = np.concatenate(data_im_es[lim_eval:lim_test]).astype(None)
gt_flat_test_es = np.concatenate(data_gt_es[lim_eval:lim_test]).astype(None)

im_flat_test_ed = np.concatenate(data_im_ed[lim_eval:lim_test]).astype(None)
gt_flat_test_ed = np.concatenate(data_gt_ed[lim_eval:lim_test]).astype(None)


#%% Load Model
PATH_model_es = "C:/Users/katrine/Documents/Universitet/Speciale/Trained_Unet_CE_sys_nor20.pt"
PATH_model_ed = "C:/Users/katrine/Documents/Universitet/Speciale/Trained_Unet_CE_dia_nor_20e.pt"

#PATH_model_es = '/Users/michalablicher/Desktop/Trained_Unet_CE_sys_nor20.pt'
#PATH_model_ed = '/Users/michalablicher/Desktop/Trained_Unet_CE_dia_nor.pt'

# Load
unet_es = torch.load(PATH_model_es, map_location=torch.device('cpu'))
unet_ed = torch.load(PATH_model_ed, map_location=torch.device('cpu'))

unet_es.eval()
out_trained_es = unet_es(Tensor(im_flat_test_es))
out_image_es   = out_trained_es["softmax"]

unet_ed.eval()
out_trained_ed = unet_ed(Tensor(im_flat_test_ed))
out_image_ed   = out_trained_ed["softmax"]

#%% One hot encoding
seg_met_dia = np.argmax(out_image_ed.detach().numpy(), axis=1)

seg_dia = torch.nn.functional.one_hot(torch.as_tensor(seg_met_dia), num_classes=4).detach().numpy()
ref_dia = torch.nn.functional.one_hot(Tensor(gt_flat_test_ed).to(torch.int64), num_classes=4).detach().numpy()

seg_met_sys = np.argmax(out_image_es.detach().numpy(), axis=1)

seg_sys = torch.nn.functional.one_hot(torch.as_tensor(seg_met_sys), num_classes=4).detach().numpy()
ref_sys = torch.nn.functional.one_hot(Tensor(gt_flat_test_es).to(torch.int64), num_classes=4).detach().numpy()


#%% E-map
import scipy.stats

emap = np.zeros((out_image_ed.shape[0],out_image_ed.shape[2],out_image_ed.shape[3]))

for i in range(0, emap.shape[0]):

    out_img = (out_image_ed[i,:,:].detach().numpy())
    entropy2 = scipy.stats.entropy(out_img)
    
    # Normalize 
    m_entropy   = np.max(entropy2)
    entropy     = entropy2/m_entropy
    emap[i,:,:] = entropy

emap = np.expand_dims(emap, axis=1)
#%% Plot
image = 7

plt.figure(dpi=2000)
plt.suptitle('Input and output of ResNet before training')
plt.subplot(2,3,1)
plt.imshow(im_flat_test_ed[image,0,:,:])
plt.ylabel('Input', fontsize=12)

plt.subplot(2,3,2)
plt.imshow(seg_met_dia[image,:,:])
plt.subplots_adjust(hspace = 0.05, wspace = 0.4)

plt.subplot(2,3,3)
plt.imshow(emap[image,0,:,:])

#% Wrap all inputs together
im     = Tensor(im_flat_test_ed)
umap   = Tensor(emap)
seg    = Tensor(np.expand_dims(seg_met_dia, axis=1))

input_concat = torch.cat((im,umap,seg), dim=1)

out    = model(input_concat)
output = out['softmax'].detach().numpy()

plt.subplot(2,3,4)
plt.imshow(output[image,0,:,:])
plt.ylabel('output', fontsize=12)
plt.subplot(2,3,5)
plt.imshow(output[image,1,:,:])

#%% Setting up training loop
# OBS DECREASED LEARNING RATE AND EPSILON ADDED TO OPTIMIZER

LEARNING_RATE = 0.0001 # 
criterion     = nn.CrossEntropyLoss() 
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

num_epoch = 3
print('Number of epochs = ',num_epoch)

#%% Prep data
T = np.expand_dims(T_j, axis=1)
T = T[0:input_concat.shape[0],:,:]

input_concat_train = input_concat[0:30,:,:,:]
input_concat_eval = input_concat[30:,:,:,:]

T_train = T[0:30,:,:,:]
T_eval = T[30:,:,:,:]

#%% Training
losses = []
losses_eval = []


trainloader = input_concat_train

for epoch in range(num_epoch):  # loop over the dataset multiple times
    
    model.train()
    print('Epoch train =',epoch)
    train_loss = []
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        #inputs, labels = data
        inputs = input_concat_train
        #inputs = inputs.cuda()
        labels = Tensor(T_train)
        #labels = labels.cuda()
        print('i=',i)
        
        # wrap them in Variable
        #inputs, labels = Variable(inputs, requires_grad=True), Variable(labels, requires_grad=True)
        inputs, labels = Variable(inputs), Variable(labels)
        labels = torch.argmax(labels, dim=1)
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
        train_loss.append(loss.item()) #.detach().cpu().numpy()
        
    losses.append(np.mean(train_loss)) # This is normalised by batch size
    train_loss = []

    model.eval()
    batch_eval_loss = []
    for i, data in enumerate(input_concat_eval, 0):
        # get the inputs
        #inputs, labels = data
        inputs = input_concat_eval
        #inputs = inputs.cuda()
        labels = Tensor(T_eval)
        #labels = labels.cuda()
        print('i=',i)
        
        # wrap them in Variable
        #inputs, labels = Variable(inputs, requires_grad=True), Variable(labels, requires_grad=True)
        inputs, labels = Variable(inputs), Variable(labels)
        labels = torch.argmax(labels, dim=1)
        labels = labels.long()
        # Clear the gradients
        optimizer.zero_grad()
       
        # Forward Pass
        output = model(inputs)     
        output = output["log_softmax"]
        
        # Find loss
        loss_eval = criterion(output, labels)
        
        # Calculate gradients
        loss_eval.backward()
        # Update Weights
        optimizer.step()
        # Calculate loss
        batch_eval_loss.append(loss_eval.item())#.detach().cpu().numpy()
        
    losses_eval.append(np.mean(batch_eval_loss)) # This is normalised by batch size
    batch_eval_loss = []
    
print('Finished Training + Evaluation')

#%% Plot loss curve
epochs = np.arange(len(losses))
epochs_eval = np.arange(len(losses_eval))

plt.figure(dpi=200)
plt.plot(epochs + 1 , losses, 'b', label='Training Loss')
plt.plot(epochs_eval + 1 , losses_eval, 'r', label='Validation Loss')
plt.xticks(np.arange(1,num_epoch+1, step = 1))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc="upper right")
plt.title("Loss function")
#plt.savefig('/home/michala/Speciale2021/Speciale2021/Trained_Unet_CE_dia_loss_20.png')

#%% Visualize output from detection network

out_test    = model(input_concat)
output_test = out_test['softmax'].detach().numpy()

image = 30

plt.subplot(1,2,1)
plt.imshow(output_test[image,0,:,:])
plt.title('Prob. of no seg. failure')
plt.colorbar(fraction=0.05)
plt.subplots_adjust(hspace = 0.05, wspace = 0.4)

plt.subplot(1,2,2)
plt.imshow(output_test[image,1,:,:])
plt.title('Prob. of seg. failure')
plt.colorbar(fraction=0.05)

#%% Upsample

test_im = Tensor(np.expand_dims(output_test[30:32,1,:,:],axis=0))
up = nn.Upsample((128,128), mode='bilinear', align_corners=True)

up_im = up(test_im)
print(np.unique(up_im))

plt.imshow(up_im[0,1,:,:])
plt.imshow(im_flat_test_ed[31,0,:,:], alpha= 0.3)