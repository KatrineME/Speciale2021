#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 12:52:25 2021

@author: michalablicher
"""
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

if torch.cuda.is_available():
    # Tensor = torch.cuda.FloatTensor
    device = 'cuda'
else:
    # Tensor = torch.FloatTensor
    device = 'cpu'
torch.cuda.manual_seed_all(808)


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
    #import torchsummary

    n_channels = 3  # 3
    n_classes  = 2
    model  = CombinedRSN(BasicBlock, channels=(16, 32, 64, 128), n_channels_input=n_channels, n_classes=n_classes, drop_prob=0.3)
    #model = SimpleRSN(BasicBlock, channels=(16, 32, 64, 128), n_channels_input=n_channels, n_classes=n_classes, drop_prob=0.5)
    model.cuda()
    #torchsummary.summary(model, (n_channels, 80, 80))
    
#%% Specify directory
os.chdir("/home/michala/Speciale2021/Speciale2021/Speciale2021/Speciale2021/") 

from load_data_gt_im_sub import load_data_sub
"""
data_im_es_DCM,  data_gt_es_DCM  = load_data_sub('GPU','Systole','DCM')
data_im_es_HCM,  data_gt_es_HCM  = load_data_sub('GPU','Systole','HCM')
data_im_es_MINF, data_gt_es_MINF = load_data_sub('GPU','Systole','MINF')
data_im_es_NOR,  data_gt_es_NOR  = load_data_sub('GPU','Systole','NOR')
data_im_es_RV,   data_gt_es_RV   = load_data_sub('GPU','Systole','RV')
"""
data_im_ed_DCM,  data_gt_ed_DCM  = load_data_sub('GPU','Diastole','DCM')
data_im_ed_HCM,  data_gt_ed_HCM  = load_data_sub('GPU','Diastole','HCM')
data_im_ed_MINF, data_gt_ed_MINF = load_data_sub('GPU','Diastole','MINF')
data_im_ed_NOR,  data_gt_ed_NOR  = load_data_sub('GPU','Diastole','NOR')
data_im_ed_RV,   data_gt_ed_RV   = load_data_sub('GPU','Diastole','RV')


#%% BATCH GENERATOR
num_train_sub = 16 
num_eval_sub  = num_train_sub + 2
num_test_sub  = num_eval_sub + 2
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



#%% Load U-NET
#% BayesUNet
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
    
#%% Load Model
#PATH_model_es = "C:/Users/katrine/Documents/Universitet/Speciale/Trained_Unet_CE_sys_nor20.pt"
#PATH_model_ed = "C:/Users/katrine/Documents/Universitet/Speciale/Trained_Unet_CE_dia_nor_20e.pt"

#PATH_model_es = '/Users/michalablicher/Desktop/Trained_Unet_CE_sys_big_batch_100_2.pt'
#PATH_model_ed = '/Users/michalablicher/Desktop/Trained_Unet_CE_dia_big_batch_100_2.pt'

#PATH_model_es = '/home/michala/Speciale2021/Speciale2021/Trained_Unet_CE_sys_sub_batch_100.pt'
PATH_model_ed = '/home/michala/Speciale2021/Speciale2021/Trained_Unet_CE_dia_sub_batch_100.pt'

# Load
#unet_es = torch.load(PATH_model_es, map_location=torch.device('cuda'))
unet_ed = torch.load(PATH_model_ed, map_location=torch.device('cuda'))

#im_flat_test_es = im_flat_test_es.cuda()

#unet_es.eval()
#out_trained_es = unet_es(Tensor(im_test_es_sub).cuda())
#out_image_es   = out_trained_es["softmax"]

#im_flat_test_ed = im_flat_test_ed.cuda()

unet_ed.eval()
out_trained_ed = unet_ed(Tensor(im_test_ed_sub).cuda())
out_image_ed   = out_trained_ed["softmax"]

#%% One hot encoding
seg_met_dia = np.argmax(out_image_ed.detach().cpu().numpy(), axis=1)

seg_dia = torch.nn.functional.one_hot(torch.as_tensor(seg_met_dia), num_classes=4).detach().cpu().numpy()
ref_dia = torch.nn.functional.one_hot(Tensor(gt_test_ed_sub).to(torch.int64), num_classes=4).detach().cpu().numpy()
"""
seg_met_sys = np.argmax(out_image_es.detach().cpu().numpy(), axis=1)

seg_sys = torch.nn.functional.one_hot(torch.as_tensor(seg_met_sys), num_classes=4).detach().cpu().numpy()
ref_sys = torch.nn.functional.one_hot(Tensor(gt_test_es_sub).to(torch.int64), num_classes=4).detach().cpu().numpy()

"""
#%% E-map
import scipy.stats

emap = np.zeros((out_image_ed.shape[0],out_image_ed.shape[2],out_image_ed.shape[3]))

for i in range(0, emap.shape[0]):

    out_img = (out_image_ed[i,:,:].detach().cpu().numpy())
    entropy2 = scipy.stats.entropy(out_img)
    
    # Normalize 
    m_entropy   = np.max(entropy2)
    entropy     = entropy2/m_entropy
    emap[i,:,:] = entropy

emap = np.expand_dims(emap, axis=1)
#%% Plot
#% Wrap all inputs together
im     = Tensor(im_test_ed_sub)
umap   = Tensor(emap)
seg    = Tensor(np.expand_dims(seg_met_dia, axis=1))

input_concat = torch.cat((im,umap,seg), dim=1)


out    = model(input_concat.cuda())
output = out['softmax'].detach().cpu().numpy()

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

num_epoch = 20
print('Number of epochs = ',num_epoch)
#%% Load T_j
#os.chdir("C:/Users/katrine/Documents/GitHub/Speciale2021")
#os.chdir("/Users/michalablicher/Documents/GitHub/Speciale2021")
os.chdir("/home/michala/Speciale2021/Speciale2021/Speciale2021/Speciale2021") 
from SI_func_mic import SI_set

lim_eval = 1
lim_test = 1
T_j = SI_set('GPU', 'dia')


#%% Prep data
T = np.expand_dims(T_j, axis=1)

print('T', T.shape)
train_amount = 34

input_concat_train = input_concat[0:train_amount,:,:,:]
input_concat_eval  = input_concat[train_amount:,:,:,:]
print('input', input_concat.shape)


print('train shape', input_concat_train.shape)
print('eval shape', input_concat_eval.shape)
T_train = Tensor(T[0:train_amount,:,:,:])
T_eval  = T[train_amount:,:,:,:]

print('T_train', T_train.shape)
print('T_eval', T_eval.shape)

#%% Training
train_losses = []
eval_losses  = []
eval_loss    = 0.0
train_loss   = 0.0


trainloader = input_concat_train

for epoch in range(num_epoch):  # loop over the dataset multiple times
    
    model.train()
    print('Epoch train =',epoch)
    for i, data_train in enumerate(trainloader, 0):
        # get the inputs
        #inputs, labels = data
        inputs = input_concat_train
        inputs = inputs.cuda()
        labels = Tensor(T_train)
        labels = labels.cuda()
        #print('i=',i)
        
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
        train_loss += loss.item()
        #train_loss.append(loss.item()) #.detach().cpu().numpy()
        
    train_losses.append(train_loss/data_train.shape[0]) # This is normalised by batch size
    train_loss = 0.0

    model.eval()
    for i, data_eval in enumerate(input_concat_eval, 0):
        # get the inputs
        #inputs, labels = data
        inputs = input_concat_eval
        inputs = inputs.cuda()
        labels = Tensor(T_eval)
        labels = labels.cuda()
        #print('i=',i)
        
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
        eval_loss += loss.item()#.detach().cpu().numpy()
        
    eval_losses.append(eval_loss/data_eval.shape[0]) # This is normalised by batch size
    eval_loss = 0.0
    

print('Finished Training + Evaluation')

#%% Plot loss curve
epochs = np.arange(len(train_losses))
epochs_eval = np.arange(len(eval_losses))

plt.figure(dpi=200)
plt.plot(epochs + 1 , train_losses, 'b', label='Training Loss')
plt.plot(epochs_eval + 1 , eval_losses, 'r', label='Validation Loss')
plt.xticks(np.arange(1,num_epoch+1, step = 1))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc="upper right")
plt.title("Loss function")
plt.savefig('/home/michala/Speciale2021/Speciale2021/Trained_detection.png')


#%% Save model
PATH_model = "/home/michala/Speciale2021/Speciale2021/Trained_Det_dia.pt"
PATH_state = "/home/michala/Speciale2021/Speciale2021/Trained_Det_dia.pt"

#PATH_model = "/home/katrine/Speciale2021/Speciale2021/Trained_Unet_CE_dia.pt"
#PATH_state = "/home/katrine/Speciale2021/Speciale2021/Trained_Unet_CE_dia_state.pt"

torch.save(unet, PATH_model)
torch.save(unet.state_dict(), PATH_state)




