#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 15:52:00 2021

@author: michalablicher
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 10:24:12 2021

@author: michalablicher
"""
import torch
import torch.nn as nn
import numpy as np
import math
import torchsummary
import torch.utils.model_zoo as model_zoo
from os import path
BatchNorm = nn.BatchNorm2d
DropOut = nn.Dropout2d

#%% Create Detection Network (sResNet)
webroot = 'https://tigress-web.princeton.edu/~fy/drn/models/'

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'drn-c-26': webroot + 'drn_c_26-ddedf421.pth',
    'drn-c-42': webroot + 'drn_c_42-9d336e8c.pth',
    'drn-c-58': webroot + 'drn_c_58-0a53a92c.pth',
    'drn-d-22': webroot + 'drn_d_22-4bd2f8ea.pth',
    'drn-d-38': webroot + 'drn_d_38-eebb45f0.pth',
    'drn-d-54': webroot + 'drn_d_54-0e0534ff.pth',
    'drn-d-105': webroot + 'drn_d_105-12b40979.pth'
}


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
        #print("DRN - Parameter info {} {} {} {} {}".format(out_map, self.out_dim, out_middle, arch, mc_dropout))

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


def drn_d_22(pretrained=False, **kwargs):
    model = DRN(BasicBlock, [1, 1, 2, 2, 2, 2, 1, 1], arch='D', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-d-22']))
    return model


class DRNDetect(nn.Module):
    def __init__(self, model, classes, pretrained_model=None):
        super(DRNDetect, self).__init__()

        self.base = nn.Sequential(*list(model.children())[:-2])
        self.detect = nn.Conv2d(model.out_dim, classes, kernel_size=1)
        self.softmax_layer = nn.Softmax(dim=1)
        self.log_softmax_layer = nn.LogSoftmax(dim=1)
        self.detect.apply(weights_init)

    def forward(self, x):
        x = self.base(x)
        x = self.detect(x)
        return {'log_softmax': self.log_softmax_layer(x), 'softmax': self.softmax_layer(x)}


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

    n_channels = 2
    n_classes = 2
    model_resnet = CombinedRSN(BasicBlock, channels=(16, 32, 64, 128), n_channels_input=n_channels, n_classes=2, drop_prob=0.5)
    #model_resnet = SimpleRSN(BasicBlock, channels=(16, 32, 64, 128), n_channels_input=n_channels, n_classes=n_classes, drop_prob=0.5)
    #model.cuda()
    torchsummary.summary(model_resnet, (n_channels, 80, 80))


#%% Prepare data for testing
# Load new image for testing
cwd = os.getcwd()
#os.chdir("C:/Users/katrine/Documents/Universitet/Speciale/ACDC_training_data/training/patient008")
os.chdir('/Users/michalablicher/Desktop/training/patient051')

nimg = nib.load('patient051_frame01.nii.gz')
img  = nimg.get_fdata()
im_slices = img.shape[2]

#%% Prepare image and run through model
# Crop image
centercrop     = torchvision.transforms.CenterCrop((128,128))
centercrop_img = Tensor(np.zeros((128,128,im_slices)))

for i in range(0,im_slices):
    centercrop_img[:,:,i] = centercrop(Tensor(img[:,:,i]))

in_image = np.expand_dims(centercrop_img,0)
in_image = Tensor(in_image).permute(3,0,1,2)

out_trained = model(in_image)
out_image = out_trained["softmax"]


#%% Create emaps
emap = np.zeros((10,128,128))
import scipy.stats
for i in range(0, out_image.shape[0]):
    plt.suptitle('E-maps')
    plt.subplot(3,5,i+1)
    plt.subplots_adjust(hspace = 0.25, wspace = 0.4)
    out_img = (out_image[i,:,:].detach().numpy())
    entropy2 = scipy.stats.entropy(out_img)
    plt.imshow(entropy2) 
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    plt.title('Slice %i' %i, fontsize =12) 
    emap[i,:,:] = entropy2
 
#%% Prepare data for detection network
test = Tensor(np.expand_dims(in_image[test_slice,:,:,:],0))
emap_test = Tensor(np.expand_dims(emap[test_slice,:,:],0))
emap_test_2 = Tensor(np.expand_dims(emap_test[:,:,:],0))

tog = torch.cat((test,emap_test_2),dim=1)

#%% Run detetcion network and print output
model_test = model_resnet(tog)
test_out = model_test["softmax"]

plt.subplot(1,2,1)
plt.imshow(test_out[0,0,:,:].detach().numpy())   
plt.subplot(1,2,2)
plt.imshow(test_out[0,1,:,:].detach().numpy())  

#%% Setting up training loop
# OBS DECREASED LEARNING RATE AND EPSILON ADDED TO OPTIMIZER

import torch.optim as optim
from torch.autograd  import Variable
#from sklearn.metrics import brier_score_loss

LEARNING_RATE = 0.0001 # 
criterion    = nn.NLLLoss() 
#criterion     = nn.BCELoss()
#criterion     = SoftDice
#criterion     = brier_score_loss()

# weight_decay is equal to L2 regularizationst
optimizer = optim.Adam(model_resnet.parameters(), lr=LEARNING_RATE, eps=1e-04, weight_decay=1e-4)
# torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

# and a learning rate scheduler which decreases the learning rate by 10x every 3 epochs
#lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
#                                               step_size=3,
#                                               gamma=0.1)

num_epoch = 2

#%% Training
losses = []
losses_eval = []
trainloader = tog

for epoch in range(num_epoch):  # loop over the dataset multiple times
    
    model.train()
    print('Epoch train =',epoch)
    train_loss = 0.0  
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        #inputs, labels = data
        inputs = tog
        labels = labels.long
        #print('i=',i)

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)
        labels = labels.long()

        # Clear the gradients
        optimizer.zero_grad()

        # Forward Pass
        output = model_resnet(inputs)     
        output = output["log_softmax"]
        
        # Find loss
        loss = criterion(output, labels)
        # Calculate gradients
        loss.backward()
        # Update Weights
        optimizer.step()
        
        # Calculate loss
        train_loss += loss.item()
        losses.append(train_loss)
        (epoch + 1, i + 1, train_loss)
        train_loss = 0.0
     


print('Finished Training + Evaluation')
        
#%% Plot loss curves

epochs = np.arange(len(losses))
epochs_eval = np.arange(len(losses_eval))
plt.figure(dpi=200)
plt.plot(epochs, losses, 'b', label='Training Loss')
plt.plot(epochs_eval, losses_eval, 'r', label='Validation Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend(loc="upper right")
plt.title("Loss function")
plt.show()


#%%
def get_loss(self, log_pred_probs, lbls, pred_probs=None):
        """

        :param log_pred_probs: LOG predicted probabilities [batch_size, 2, w * h]
        :param lbls: ground truth labels [batch_size, w * h]
        :param pred_probs: [batch_size, 2, w * h]
        :return: torch scalar
        """
        # print("INFO - get_loss - log_pred_probs.shape, lbls.shape ", log_pred_probs.shape, lbls.shape)
        # NOTE: this was a tryout (not working) for hard negative mining
        # batch_loss_indices = RegionDetector.hard_negative_mining(pred_probs, lbls)
        # b_loss_idx_preds = batch_loss_indices.unsqueeze(1).expand_as(log_pred_probs)
        # The input given through a forward call is expected to contain log-probabilities of each class
        b_loss = self.loss_function(log_pred_probs, lbls)

        # pred_probs last 2 dimensions need to be merged because lbls has shape [batch_size, w, h ]
        pred_probs = pred_probs.view(pred_probs.size(0), 2, -1)
        fn_soft = pred_probs[:, 0] * lbls.float()
        # fn_nonzero = torch.nonzero(fn_soft.data).size(0)
        batch_size = pred_probs.size(0)
        fn_soft = torch.sum(fn_soft) * 1 / float(batch_size)
        # same for false positive
        ones = torch.ones(lbls.size())#.cuda()
        fp_soft = (ones - lbls.float()) * pred_probs[:, 1]
        # fp_nonzero = torch.nonzero(fp_soft).size(0)
        fp_soft = torch.sum(fp_soft) * 1 / float(batch_size)
        # print(b_loss.item(), (self.fn_penalty_weight * fn_soft + self.fp_penalty_weight * fp_soft).item())
        b_loss = b_loss + self.fn_penalty_weight * fn_soft + self.fp_penalty_weight * fp_soft

        return b_loss
    
    
#%%    
for epoch in range(num_epoch):  # loop over the dataset multiple times
    
    model_resnet.train()
    print('Epoch train =',epoch)
    train_loss = 0.0  
    for i, data in enumerate(trainloader, 0):
    #def train(self, images, ref_labels, y_labels_seg):
        model_resnet.train()
        optimizer.zero_grad()
        ref_labels = Tensor(gt_flat_train)
        output = model_resnet(tog)
        batch_size, channels, _, _ = output["log_softmax"].size()
        loss = get_loss(output["log_softmax"].view(batch_size, channels, -1), ref_labels, pred_probs=output["softmax"])
        y_labels_seg = torch.LongTensor(torch.from_numpy(y_labels_seg).long())#.cuda()
        seg_loss = loss_function(output["log_softmax_y"].view(batch_size, channels, -1), y_labels_seg.view(batch_size, -1))
        loss = loss + 5 * seg_loss
        loss.backward()
        _train_iter += 1
        optimizer.step()
        scheduler.step()
        current_training_loss = loss.detach().numpy() #loss.detach().cpu().numpy()
        training_losses.append(current_training_loss)
        
        
        
        
        
