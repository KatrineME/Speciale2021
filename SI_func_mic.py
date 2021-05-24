#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 12:45:28 2021

@author: michalablicher
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 14:03:20 2021

@author: katrine
"""
def SI_set(user, phase):

    #%% Load packages
    import torch
    import os
    import numpy   as np

    from torch import nn
    from torch import Tensor
    
    if torch.cuda.is_available():
        # Tensor = torch.cuda.FloatTensor
        device = 'cuda'
    else:
        # Tensor = torch.FloatTensor
        device = 'cpu'
    torch.cuda.manual_seed_all(808)
        
    #%% Specify directory
    if user == 'K':
        os.chdir("C:/Users/katrine/Documents/GitHub/Speciale2021")
    elif user == 'GPU':
        os.chdir("/home/michala/Speciale2021/Speciale2021/Speciale2021/Speciale2021")                      # Server directory michala
    else:
        os.chdir('/Users/michalablicher/Documents/GitHub/Speciale2021')
    
    # Load data function
    from load_data_gt_im_sub import load_data_sub
    
    if phase == 'sys':
       data_im_es_DCM,  data_gt_es_DCM  = load_data_sub(user,'Systole','DCM')
       data_im_es_HCM,  data_gt_es_HCM  = load_data_sub(user,'Systole','HCM')
       data_im_es_MINF, data_gt_es_MINF = load_data_sub(user,'Systole','MINF')
       data_im_es_NOR,  data_gt_es_NOR  = load_data_sub(user,'Systole','NOR')
       data_im_es_RV,   data_gt_es_RV   = load_data_sub(user,'Systole','RV')
    else:
        data_im_es_DCM,  data_gt_es_DCM  = load_data_sub(user,'Diastole','DCM')
        data_im_es_HCM,  data_gt_es_HCM  = load_data_sub(user,'Diastole','HCM')
        data_im_es_MINF, data_gt_es_MINF = load_data_sub(user,'Diastole','MINF')
        data_im_es_NOR,  data_gt_es_NOR  = load_data_sub(user,'Diastole','NOR')
        data_im_es_RV,   data_gt_es_RV   = load_data_sub(user,'Diastole','RV')
    

#%% BATCH GENERATOR
    """
    num_train_sub = 16 
    num_eval_sub = num_train_sub + 2
    num_test_sub = num_eval_sub + 2
    
    
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
    """
    
    num_train_sub = 16 
    num_eval_sub  = num_train_sub + 1
    
    
    
    num_train_res  = num_eval_sub + 2
    num_test_res  = num_train_res + 1
    
    im_train_es_res = np.concatenate((np.concatenate(data_im_es_DCM[num_eval_sub:num_train_res]).astype(None),
                                      np.concatenate(data_im_es_HCM[num_eval_sub:num_train_res]).astype(None),
                                      np.concatenate(data_im_es_MINF[num_eval_sub:num_train_res]).astype(None),
                                      np.concatenate(data_im_es_NOR[num_eval_sub:num_train_res]).astype(None),
                                      np.concatenate(data_im_es_RV[num_eval_sub:num_train_res]).astype(None)))
    
    gt_train_es_res = np.concatenate((np.concatenate(data_gt_es_DCM[num_eval_sub:num_train_res]).astype(None),
                                      np.concatenate(data_gt_es_HCM[num_eval_sub:num_train_res]).astype(None),
                                      np.concatenate(data_gt_es_MINF[num_eval_sub:num_train_res]).astype(None),
                                      np.concatenate(data_gt_es_NOR[num_eval_sub:num_train_res]).astype(None),
                                      np.concatenate(data_gt_es_RV[num_eval_sub:num_train_res]).astype(None)))
    


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
    
    #%% Load model
    if user == 'K':
        PATH_model_es = "C:/Users/katrine/Documents/Universitet/Speciale/Trained_Unet_CE_sys_sub_batch_100.pt"
        PATH_model_ed = "C:/Users/katrine/Documents/Universitet/Speciale/Trained_Unet_CE_dia_sub_batch_100.pt"
    elif user == 'GPU':
        PATH_model_es = "/home/michala/Speciale2021/Speciale2021/Trained_Unet_CE_sys_sub_batch_100.pt"  
        PATH_model_ed = "/home/michala/Speciale2021/Speciale2021/Trained_Unet_CE_dia_sub_batch_100.pt"                    # Server directory michala
    else:
        PATH_model_es = '/Users/michalablicher/Desktop/Trained_Unet_CE_sys_sub_batch_100.pt'
        PATH_model_ed = '/Users/michalablicher/Desktop/Trained_Unet_CE_dia_sub_batch_100.pt'
        
    
    if phase == 'sys':
        unet = torch.load(PATH_model_es, map_location=torch.device(device))
    else:
        unet = torch.load(PATH_model_ed, map_location=torch.device(device))
        
    #%% Running  models 
    # SYSTOLIC
    unet.eval()
    output_unet= unet(Tensor(im_train_es_res))
    output_unet= output_unet["softmax"]
    print('res shape', im_train_es_res.shape)
    #output_unet_es_eval = unet_es(Tensor(im_flat_eval_es))
    #output_unet_es_eval = output_unet_es_eval["softmax"]
    
    #output_unet_es_test = unet_es(Tensor(im_flat_test_es))
    #output_unet_es_test = output_unet_es_test["softmax"]
    
    # DIASTOLIC
    #unet_ed.eval()
    #output_unet_ed_train = unet_es(Tensor(im_flat_train_ed))
    #output_unet_ed_train = output_unet_ed_train["softmax"]
    
    #output_unet_ed_eval = unet_es(Tensor(im_flat_eval_ed))
    #output_unet_ed_eval = output_unet_ed_eval["softmax"]
    
    #output_unet_ed_test = unet_es(Tensor(im_flat_test_ed))
    #output_unet_ed_test = output_unet_ed_test["softmax"]
    
    #%% Onehot encode class channels
    gt_es_oh_train = torch.nn.functional.one_hot(Tensor(gt_train_es_res).to(torch.int64), num_classes=4).detach().cpu().numpy().astype(np.bool)
    
    # Argmax
    seg_met_sys_train = np.argmax(output_unet.detach().cpu().numpy(), axis=1)
    #seg_met_dia_train = np.argmax(output_unet_ed_train.detach().numpy(), axis=1)
    
    #seg_met_sys_eval = np.argmax(output_unet_es_eval.detach().numpy(), axis=1)
    #seg_met_dia_eval = np.argmax(output_unet_ed_eval.detach().numpy(), axis=1)
    
    #seg_met_sys_test = np.argmax(output_unet_es_test.detach().numpy(), axis=1)
    #seg_met_dia_test = np.argmax(output_unet_ed_test.detach().numpy(), axis=1)
    
    #One hot encoding
    seg_sys_train = torch.nn.functional.one_hot(torch.as_tensor(seg_met_sys_train), num_classes=4).detach().cpu().numpy()
    #seg_dia_train = torch.nn.functional.one_hot(torch.as_tensor(seg_met_dia_train), num_classes=4).detach().numpy()
    
    #seg_sys_eval = torch.nn.functional.one_hot(torch.as_tensor(seg_met_sys_eval), num_classes=4).detach().numpy()
    #seg_dia_eval = torch.nn.functional.one_hot(torch.as_tensor(seg_met_dia_eval), num_classes=4).detach().numpy()
    
    #seg_sys_test = torch.nn.functional.one_hot(torch.as_tensor(seg_met_sys_test), num_classes=4).detach().numpy()
    #seg_dia_test = torch.nn.functional.one_hot(torch.as_tensor(seg_met_dia_test), num_classes=4).detach().numpy()
    

    #%% Distance transform maps
    
    #Specify directory
    if user == 'K':
        os.chdir("C:/Users/katrine/Documents/GitHub/Speciale2021")
    elif user == 'GPU':
        os.chdir("/home/michala/Speciale2021/Speciale2021/Speciale2021/Speciale2021")                      # Server directory michala
    else:
        os.chdir('/Users/michalablicher/Documents/GitHub/Speciale2021')
        
    from SI_error_func import dist_trans, cluster_min
    
    error_margin_inside  = 2
    error_margin_outside = 3
    
    # Distance transform map
    dt_es_train = dist_trans(gt_es_oh_train, error_margin_inside,error_margin_outside)
    #dt_ed_train = dist_trans(gt_ed_oh_train, error_margin_inside,error_margin_outside)
    

    #dt_es_eval = dist_trans(gt_es_oh_eval , error_margin_inside,error_margin_outside)
    #dt_ed_eval = dist_trans(gt_ed_oh_eval , error_margin_inside,error_margin_outside)
    
    #dt_es_test = dist_trans(gt_es_oh_test, error_margin_inside,error_margin_outside)
    #dt_ed_test = dist_trans(gt_ed_oh_test, error_margin_inside,error_margin_outside)
    
    #%% filter cluster size
    cluster_size = 10
    
    #dia_new_label_train = cluster_min(seg_dia_train, gt_ed_oh_train, cluster_size)
    sys_new_label_train = cluster_min(seg_sys_train, gt_es_oh_train, cluster_size)


    #dia_new_label_eval = cluster_min(seg_dia_eval, gt_ed_oh_eval, cluster_size)
    #sys_new_label_eval = cluster_min(seg_sys_eval, gt_es_oh_eval, cluster_size)    
    
    #dia_new_label_test = cluster_min(seg_dia_test, gt_ed_oh_test, cluster_size)
    #sys_new_label_test = cluster_min(seg_sys_test, gt_es_oh_test, cluster_size)
    #%% Apply both cluster size and dt map 
    roi_es_train = np.zeros((dt_es_train.shape))
    #roi_ed_train = np.zeros((dt_ed_train.shape))
    
    #roi_es_eval = np.zeros((dt_es_eval.shape))
    #roi_ed_eval = np.zeros((dt_ed_eval.shape))
    
    #roi_es_test = np.zeros((dt_es_test.shape))
    #roi_ed_test = np.zeros((dt_ed_test.shape))
    
    for i in range(0, dt_es_train.shape[0]):
        for j in range(0, dt_es_train.shape[3]):
            roi_es_train[i,:,:,j] = np.logical_and(dt_es_train[i,:,:,j], sys_new_label_train[i,:,:,j])
            #roi_ed_train[i,:,:,j] = np.logical_and(dt_ed_train[i,:,:,j], dia_new_label_train[i,:,:,j])

    
    #for i in range(0, dt_es_eval.shape[0]):
    #    for j in range(0, dt_es_eval.shape[3]):
    #        roi_es_eval[i,:,:,j] = np.logical_and(dt_es_eval[i,:,:,j], sys_new_label_eval[i,:,:,j])
    #        roi_ed_eval[i,:,:,j] = np.logical_and(dt_ed_eval[i,:,:,j], dia_new_label_eval[i,:,:,j])

    #for i in range(0, dt_es_test.shape[0]):
    #    for j in range(0, dt_es_test.shape[3]):
    #        roi_es_test[i,:,:,j] = np.logical_and(dt_es_test[i,:,:,j], sys_new_label_test[i,:,:,j])
    #        roi_ed_test[i,:,:,j] = np.logical_and(dt_ed_test[i,:,:,j], dia_new_label_test[i,:,:,j])
                
    #%% Sample patches
    patch_size = 8
    patch_grid = int(roi_es_train.shape[1]/patch_size)
    
    lin    = np.linspace(0,roi_es_train.shape[1]-patch_size,patch_grid).astype(int)
    
   
    # Preallocate
    _temp  = np.zeros((patch_grid,patch_grid))
    lin    = np.linspace(0,roi_es_train.shape[1]-patch_size,patch_grid).astype(int)
    _ctemp = np.zeros((patch_grid,patch_grid,roi_es_train.shape[3]))
    T_j    = np.zeros((roi_es_train.shape[0],patch_grid,patch_grid,roi_es_train.shape[3]))
    

    for j in range (0,roi_es_train.shape[0]):
        for c in range(0,4):
            for pp in range(0,16):
                for p, i in enumerate(lin):
                    _temp[pp,p] = np.count_nonzero(roi_es_train[j,lin[pp]:lin[pp]+8 , i:i+8, c])
                    #_temp[pp,p] = np.sum(~np.isnan(roi_es_train[j,lin[pp]:lin[pp]+8 , i:i+8, c]))
            _ctemp[:,:,c] = _temp
        T_j[j,:,:,:] = _ctemp
    
    
    # BACKGROUND SEG FAILURES ARE REMOVED
    T_j = T_j[:,:,:,1:] 
    
    # Summing all tissue channels together
    T_j = np.sum(T_j, axis = 3)

    # Plot a final patch
    # Binarize
    T_j[T_j >= 1 ] = 1

    
    # EVALUATION
    #T_j_ed_eval    = np.zeros((roi_ed_eval.shape[0], patch_grid, patch_grid, roi_ed_eval.shape[3]))
    #T_j_es_eval    = np.zeros((roi_es_eval.shape[0], patch_grid, patch_grid, roi_es_eval.shape[3]))
    
    
    #for j in range (0,roi_es_eval.shape[0]):
    #    for c in range(0,4):
    #        for pp in range(0,16):
    #            for p, i in enumerate(lin):
    #                _temp_ed[pp,p] = np.count_nonzero(~np.isnan(roi_ed_eval[j,lin[pp]:lin[pp]+8 , i:i+8, c]))
    #                _temp_es[pp,p] = np.count_nonzero(~np.isnan(roi_es_eval[j,lin[pp]:lin[pp]+8 , i:i+8, c]))
    #                
    #        _ctemp_ed[:,:,c] = _temp_ed
    #        _ctemp_es[:,:,c] = _temp_es
    #        
    #    T_j_ed_eval[j,:,:,:] = _ctemp_ed
    #    T_j_es_eval[j,:,:,:] = _ctemp_es
        
    # TEST
    """T_j_ed_test    = np.zeros((roi_ed_test.shape[0], patch_grid, patch_grid, roi_ed_test.shape[3]))
    T_j_es_test    = np.zeros((roi_es_test.shape[0], patch_grid, patch_grid, roi_es_test.shape[3]))
    
    
    for j in range (0,roi_es_test.shape[0]):
        for c in range(0,4):
            for pp in range(0,16):
                for p, i in enumerate(lin):
                    _temp_ed[pp,p] = np.count_nonzero(~np.isnan(roi_ed_test[j,lin[pp]:lin[pp]+8 , i:i+8, c]))
                    _temp_es[pp,p] = np.count_nonzero(~np.isnan(roi_es_test[j,lin[pp]:lin[pp]+8 , i:i+8, c]))
                    
            _ctemp_ed[:,:,c] = _temp_ed
            _ctemp_es[:,:,c] = _temp_es
            
        T_j_ed_test[j,:,:,:] = _ctemp_ed
        T_j_es_test[j,:,:,:] = _ctemp_es
"""
#%%    
    # BACKGROUND SEG FAILURES ARE REMOVED
    #T_j_es_train = T_j[:,:,:,1:]
    """
    T_j_es_train = T_j_es_train[:,:,:,1:]
    
    T_j_ed_eval  = T_j_ed_eval[:,:,:,1:]
    T_j_es_eval  = T_j_es_eval[:,:,:,1:]
    
    T_j_ed_test  = T_j_ed_test[:,:,:,1:]
    T_j_es_test  = T_j_es_test[:,:,:,1:]
    """
    # Summing all tissue channels together
   # T_j_es_train = np.sum(T_j_es_train, axis = 3)
    """
    T_j_es_train = np.sum(T_j_es_train, axis = 3)
    
    T_j_ed_eval = np.sum(T_j_ed_eval, axis = 3)
    T_j_es_eval = np.sum(T_j_es_eval, axis = 3)
    
    T_j_ed_test = np.sum(T_j_ed_test, axis = 3)
    T_j_es_test = np.sum(T_j_es_test, axis = 3)
    """
    # Plot a final patch
    # Binarize
    #T_j_es_train[T_j_es_train >= 1 ] = 1
    #T_j_es_train[T_j_es_train >= 1 ] = 1
    """
    T_j_ed_eval[T_j_ed_eval   >= 1 ] = 1
    T_j_es_eval[T_j_es_eval   >= 1 ] = 1
    
    T_j_ed_test[T_j_ed_test   >= 1 ] = 1
    T_j_es_test[T_j_es_test   >= 1 ] = 1
"""
    return T_j #, T_j_es_train, T_j_ed_eval, T_j_es_eval, T_j_ed_test, T_j_es_test
