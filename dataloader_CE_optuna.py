#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 15:23:08 2021

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
import optuna 
from optuna.trial import TrialState
 
from torch.autograd  import Variable
from torch import nn
from torch import Tensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold


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
"""
if __name__ == "__main__":
    #import torchsummary
    unet = BayesUNet(num_classes=4, in_channels=1, drop_prob=0.1)
    unet.cuda()
    #torchsummary.summary(model, (1, 128, 128))
"""
def define_model(trial):
    drop_prob_t = trial.suggest_float("drop_prob_l", 0.0, 0.5) 
    unet = BayesUNet(num_classes=4, in_channels=1, drop_prob=drop_prob_t)
    unet.cuda()
    
    return unet

    
#%% Specify directory
cwd = os.getcwd()
#os.chdir("C:/Users/katrine/Documents/Universitet/Speciale/ACDC_training_data/training")   # Local directory katrine
#os.chdir('/Users/michalablicher/Desktop/training')     # Local directory michala
os.chdir("/home/michala/training")                      # Server directory michala


#%% Specify directory
#os.chdir("C:/Users/katrine/Documents/GitHub/Speciale2021")
#os.chdir('/Users/michalablicher/Documents/GitHub/Speciale2021')

from load_data_gt_im_sub_space import load_data_sub


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
num_train_sub = 12
#num_eval_sub = num_train_sub + 2
num_test_sub = num_train_sub + 8

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


im_test_sub = np.concatenate((np.concatenate(data_im_ed_DCM[num_train_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_im_ed_HCM[num_train_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_im_ed_MINF[num_train_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_im_ed_NOR[num_train_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_im_ed_RV[num_train_sub:num_test_sub]).astype(None)))

gt_test_sub = np.concatenate((np.concatenate(data_gt_ed_DCM[num_train_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_gt_ed_HCM[num_train_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_gt_ed_MINF[num_train_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_gt_ed_NOR[num_train_sub:num_test_sub]).astype(None),
                                  np.concatenate(data_gt_ed_RV[num_train_sub:num_test_sub]).astype(None)))


#%% Training with K-folds
def objective(trial):
      
    model_unet = define_model(trial).to(device)
    
    #optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])
    weight_decay   = trial.suggest_float("weight_decay", 1e-8, 1e-2)

    lr  = trial.suggest_float("lr",  1e-5, 1e-2)
    eps = trial.suggest_float("eps", 1e-8, 1e-2)
    
    #optimizer = getattr(optim, Adam)(model_unet.parameters(), lr=lr, eps = eps, weight_decay = weight_decay)
    optimizer = torch.optim.Adam(model_unet.parameters(), lr=lr, eps=eps, weight_decay=weight_decay) #LR 
    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=180)
    
    k_folds    = 6
    num_epochs = 50
    #num_epochs  = trial.suggest_float("num_epochs",  5, 100)
    
    loss_function = nn.CrossEntropyLoss()
    
    # For fold results
    
    # Set fixed random number seed
    torch.manual_seed(42)
    
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)
    
    # Start print
    print('--------------------------------')
    
    # Prep data for dataloader
    data_train   = Tensor((np.squeeze(im_train_sub), gt_train_sub))
    data_train_n = data_train.permute(1,0,2,3)
    dataset      = data_train_n
    batch_size   = 32
    
    fold_train_losses = []
    fold_eval_losses  = []
    fold_train_accuracy    = []
    fold_eval_accuracy     = []
    fold_train_incorrect   = []
    fold_eval_incorrect    = []
    
    #%
    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')
        
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler  = torch.utils.data.SubsetRandomSampler(test_ids)
        
        # Define data loaders for training and testing data in this fold
        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler, drop_last=True)
        eval_dataloader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_subsampler,  drop_last=True)
       
        
        # Init the neural network
        #network = unet()
        model_unet.apply(weights_init)
        
        # Initialize optimizer
        #optimizer = torch.optim.Adam(unet.parameters(), lr=0.0001, eps=1e-04, weight_decay=1e-4)
    
        #% Training
        train_losses  = []
        train_accuracy = []
        train_incorrect = []
        eval_losses   = []
        eval_accuracy  = []
        eval_incorrect = []
        eval_loss     = 0.0
        train_loss    = 0.0
        total         = 0.0
        correct       = 0.0
        incorrect     = 0.0
        total_e         = 0.0
        correct_e       = 0.0
        incorrect_e     = 0.0
    
        for epoch in range(num_epochs):  # loop over the dataset multiple times
            
            model_unet.train()
            print('Epoch train =',epoch)
            #0.0  
            for i, (train_data) in enumerate(train_dataloader):
                # get the inputs
                #print('train_data = ', train_data.shape)
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
                output = model_unet(inputs)     
                output = output["log_softmax"]
                output = torch.exp(output)
                #print('output shape = ', output.shape)
                
                # Find loss
                loss = loss_function(output, labels)
                #print('loss = ', loss)
                
                # Calculate gradients
                loss.backward()
                
                # Update Weights
                optimizer.step()
        
                # Calculate loss
                train_loss += loss.item() #.detach().cpu().numpy()
                
                # Set total and correct
                predicted  = torch.argmax(output, axis=1)
                total     += (labels.shape[0])*(128*128)
                correct   += (predicted == labels).sum().item()
                incorrect += (predicted != labels).sum().item()
            
            train_losses.append(train_loss/(i+1)) #train_data.shape[0]) # This is normalised by batch size
            #print('epoch loss = ', train_losses)
        
            #train_losses.append(np.mean(batch_loss))
            train_loss = 0.0 #[]
            
            # Print accuracy
            #print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
            train_accuracy.append(100.0 * correct / total)
            train_incorrect.append(incorrect)
            correct   = 0.0
            total     = 0.0
            incorrect = 0.0
            
            #print('train_accuracy', train_accuracy)
            #print('--------------------------------')
            #accuracy[fold] = 100.0 * (correct / total)
            
            model_unet.eval()
            print('Epoch eval=',epoch)
             
            for j, (eval_data) in enumerate(eval_dataloader):
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
                output = model_unet(inputs)     
                output = output["log_softmax"]
                output = torch.exp(output)
                # Find loss
                loss = loss_function(output, labels)
                
                # Calculate loss
                #eval_loss.append(loss.item())
                eval_loss += loss.item() #.detach().cpu().numpy()
                
                # Set total and correct
                predicted_e = torch.argmax(output, axis=1)
                total_e     += (labels.shape[0])*(128*128)
                correct_e   += (predicted_e == labels).sum().item()
                incorrect_e += (predicted_e != labels).sum().item()
                
            eval_losses.append(eval_loss/(j+1)) # This is normalised by batch size (i = 12)
            #eval_losses.append(np.mean(eval_loss))
            eval_loss = 0.0
            
            # Print accuracy
            #print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
            eval_accuracy.append(100.0 * correct_e / total_e)
            eval_incorrect.append(incorrect_e)
            #print('bf float', eval_accuracy)

            eval_accuracy_float = float(eval_accuracy[-1])
            print('float', eval_accuracy_float)
            
            trial.report(dice, epoch)
       
            correct_e   = 0.0
            total_e     = 0.0
            incorrect_e = 0.0
            
            #lr_get   = lr_scheduler.get_last_lr()[0]
            #lr_scheduler.step()
            
        fold_train_losses.append(train_losses)
        #print('fold loss = ', fold_train_losses)
        
        fold_eval_losses.append(eval_losses)
        #print('fold loss = ', fold_eval_losses)
        
        fold_train_accuracy.append(train_accuracy)
        #print('fold loss = ', fold_train_accuracy)
        
        fold_eval_accuracy.append(eval_accuracy)
        #print('fold loss = ', fold_eval_accuracy)
        
        fold_train_incorrect.append(train_incorrect)
        #print('fold loss = ', fold_train_accuracy)
        
        fold_eval_incorrect.append(eval_incorrect)
        
        #Save model for each fold
        #PATH_model = "/home/michala/Speciale2021/Speciale2021/Trained_Unet_CE_dia_optuna_fold{}.pt".format(fold)
        #PATH_model = "/home/katrine/Speciale2021/Speciale2021/Trained_Unet_CE_dia_fold{}.pt".format(fold)
        #torch.save(model_unet, PATH_model)
        
    return eval_accuracy_float


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=50000) # 50000 s = 14 h

    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        
    
    plt.figure(dpi=200)
    optuna.visualization.matplotlib.plot_contour(study, params=["lr", "eps"])
    plt.savefig('/home/michala/Speciale2021/Speciale2021/optuna_lr_eps.png')
    
    plt.figure(dpi=200)
    optuna.visualization.matplotlib.plot_contour(study, params=["lr", "drop_prob_l"])
    plt.savefig('/home/michala/Speciale2021/Speciale2021/optuna_lr_drop.png')
    
    plt.figure(dpi=200)
    optuna.visualization.matplotlib.plot_contour(study, params=["lr", "weight_decay"])
    plt.savefig('/home/michala/Speciale2021/Speciale2021/optuna_lr_wd.png')
    
    plt.figure(dpi=200)
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.savefig('/home/michala/Speciale2021/Speciale2021/importances_optuna.png')
    
    plt.figure(dpi=200)
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.savefig('/home/michala/Speciale2021/Speciale2021/history_optuna.png')



"""    
PATH_model = "/home/michala/Speciale2021/Speciale2021/Trained_Unet_CE_dia_CrossVal_optuna.pt"
#PATH_state = "/home/michala/Speciale2021/Speciale2021/Trained_Unet_CE_batch_state.pt"

#PATH_model = "/home/katrine/Speciale2021/Speciale2021/Trained_Unet_CE_dia.pt"
#PATH_state = "/home/katrine/Speciale2021/Speciale2021/Trained_Unet_CE_dia_state.pt"

torch.save(unet, PATH_model)
#torch.save(unet.state_dict

print('Model saved')   


m_fold_train_losses    = np.mean(fold_train_losses, axis = 0) 
m_fold_eval_losses     = np.mean(fold_eval_losses, axis = 0)   
m_fold_train_accuracy  = np.mean(fold_train_accuracy, axis = 0)   
m_fold_eval_accuracy   = np.mean(fold_eval_accuracy, axis = 0)   
m_fold_train_incorrect = np.mean(fold_train_incorrect, axis = 0)   
m_fold_eval_incorrect  = np.mean(fold_eval_incorrect, axis = 0)       
    
print('Finished Training + Evaluation')

#%% Plot loss curves
epochs_train = np.arange(len(train_losses))
epochs_eval  = np.arange(len(eval_losses))

plt.figure(figsize=(30, 15), dpi=200)
plt.subplot(1,3,1)
plt.plot(epochs_train + 1 , m_fold_train_losses, 'b', label = 'Training Loss')
plt.plot(epochs_eval  + 1 , m_fold_eval_losses,  'r', label = 'Validation Loss')
plt.xticks(np.arange(1, num_epochs + 1, step = 50))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc="upper right")
plt.title("Loss function")

plt.subplot(1,3,2)
plt.plot(epochs_train + 1 , m_fold_train_accuracy, 'b', label = 'Training Acc')
plt.plot(epochs_eval  + 1 , m_fold_eval_accuracy,  'r', label = 'Validation Acc')
plt.xticks(np.arange(1, num_epochs + 1, step = 50))
plt.xlabel('Epochs')
plt.ylabel('Accuracy %')
plt.legend(loc="upper right")
plt.title("Accuracy")

plt.subplot(1,3,3)
plt.plot(epochs_train + 1 , m_fold_train_incorrect, 'b', label = 'Training Acc')
plt.plot(epochs_eval  + 1 , m_fold_eval_incorrect,  'r', label = 'Validation Acc')
plt.xticks(np.arange(1, num_epochs + 1, step = 50))
plt.xlabel('Epochs')
plt.ylabel('incorrect %')
plt.legend(loc="upper right")
plt.title("Incorrect")


plt.savefig('/home/michala/Speciale2021/Speciale2021/Trained_Unet_CE_dia_CV_acc_optuna.png')
#plt.savefig('/home/katrine/Speciale2021/Speciale2021/Trained_Unet_CE_dia_loss.png')

t_res_mean = [m_fold_train_losses, m_fold_eval_losses, m_fold_train_accuracy, m_fold_eval_accuracy, m_fold_train_incorrect, m_fold_eval_incorrect] # mean loss and accuracy
t_res      = [fold_train_losses, fold_eval_losses, fold_train_accuracy, fold_eval_accuracy]         # loss and accuracy for each epoch

T = [t_res_mean, t_res] # listed together


#%% Save model
PATH_model = "/home/michala/Speciale2021/Speciale2021/Trained_Unet_CE_dia_CrossVal_optuna.pt"
#PATH_state = "/home/michala/Speciale2021/Speciale2021/Trained_Unet_CE_batch_state.pt"

#PATH_model = "/home/katrine/Speciale2021/Speciale2021/Trained_Unet_CE_dia.pt"
#PATH_state = "/home/katrine/Speciale2021/Speciale2021/Trained_Unet_CE_dia_state.pt"

torch.save(unet, PATH_model)
#torch.save(unet.state_dict(), PATH_state)

#%%
PATH_results = "/home/michala/Speciale2021/Speciale2021/Trained_Unet_CE_dia_train_results_optuna.pt"
torch.save(T, PATH_results)    

"""

