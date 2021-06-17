# -*- coding: utf-8 -*-
"""
Created on Tue May 11 13:04:26 2021

@author: katrine
"""

def load_data_sub(user, phase, diagnose):

    #% Load packages
    import os
    import scipy
    import scipy.ndimage
    import nibabel as nib
    import numpy   as np
    import matplotlib.pyplot as plt
    import glob2
    import SimpleITK as sitk
        
    from torch import Tensor
    
    # Load paths
    if user == 'K':
        os.chdir("C:/Users/katrine/Documents/Universitet/Speciale/ACDC_training_data/training")
    elif user == 'GPU':
        os.chdir("/home/michala/training")                      # Server directory michala
    else:
        os.chdir('/Users/michalablicher/Desktop/training')
        
    #frame_dia_im = np.sort(glob2.glob('patient*/**/patient*_frame01.nii.gz'))
    frame_im = np.sort(glob2.glob('patient*/**/patient*_frame*[0-9].nii.gz'))
    frame_gt = np.sort(glob2.glob('patient*/**/patient*_frame*[0-9]_gt.nii.gz'))
        
    
    if phase == 'Diastole':
        phase = np.linspace(0,len(frame_im)-2,100).astype(int)
    else:
        phase = np.linspace(1,len(frame_im)-1,100).astype(int)
        
    #% Divide frames
    frame_im = frame_im[phase]
    frame_gt = frame_gt[phase]
        
    #% Load images
    H  = 128
    W  = 128

    center = np.zeros((100,2))

    gt_crop = [] #np.zeros((H,W,100))
    im_crop = []
    
    num_case = 20  # Number of patients within each subgroup
    
    DCM  = np.linspace(0, num_case-1, num_case).astype(int)
    HCM  = np.linspace(DCM[-1]+1,  DCM[-1]+num_case,  num_case).astype(int)
    MINF = np.linspace(HCM[-1]+1,  HCM[-1]+num_case,  num_case).astype(int)
    NOR  = np.linspace(MINF[-1]+1, MINF[-1]+num_case, num_case).astype(int)
    RV   = np.linspace(NOR[-1]+1,  NOR[-1]+num_case,  num_case).astype(int)
    
    if diagnose == 'DCM':
        sub = DCM
    elif diagnose == 'HCM':
        sub = HCM
    elif diagnose == 'MINF':
        sub = MINF
    elif diagnose == 'NOR':
        sub = NOR
    else:
        sub = RV
        
    def load_itk(filename):
        itkimage = sitk.ReadImage(filename)
        image    = np.transpose(sitk.GetArrayFromImage(itkimage))
        spacing  = np.array(itkimage.GetSpacing())
        
        return image, spacing
    
    def apply_2d_zoom_3d(arr3d, spacing, new_spacing, order=1, do_blur=False, as_type=np.float32):
        """
        :param arr3d: [#slices, IH, IW]
        :param spacing: spacing has shape [#slices, IH, IW]
        :param new_vox_size: tuple(x, y)
        :param order: of B-spline
        :param do_blur: boolean (see below)
        :param as_type: return type of np array. We use this to indicate binary/integer labels which we'll round off
                        to prevent artifacts
        :return:
        """
        if len(spacing) > 2:
            spacing = spacing[int(len(spacing) - 2):]
    
        if len(new_spacing) > 2:
            new_spacing = new_spacing[int(len(new_spacing) - 2):]
    
        zoom = np.array(spacing, float) / new_spacing
        if do_blur:
            for z in range(arr3d.shape[0]):
                sigma = .25 / zoom
                arr3d[z, :, :] = scipy.ndimage.gaussian_filter(arr3d[z, :, :], sigma)
    
        resized_img = scipy.ndimage.interpolation.zoom(arr3d, tuple((1,)) + tuple(zoom), order=order)
        if as_type == np.int:
            # binary/integer labels
            resized_img = np.round(resized_img).astype(as_type)
        return resized_img
    
    for i in sub:
        """
        nimg = nib.load(frame_im[i])   # Load nii image
        img  = nimg.get_fdata()
    
        n_gt = nib.load(frame_gt[i])   # Load nii labels
        gt   = n_gt.get_fdata()
        """
        img, spacing = load_itk(frame_im[i])
        gt, _      = load_itk(frame_gt[i])
        
          
        img = Tensor(img).permute(2,0,1).detach().numpy()
        gt = Tensor(gt).permute(2,0,1).detach().numpy()
        
        original_spacing = [spacing[2],spacing[0],spacing[1]]
            
        new_spacing = np.array([original_spacing[0], 1.4, 1.4]).astype(np.float32)
        #print('New spacing = ', new_spacing)    
        
        out_im_space = apply_2d_zoom_3d(img, original_spacing, new_spacing, order=1, do_blur=False , as_type=np.float32)
        out_gt_space = apply_2d_zoom_3d(gt, original_spacing, new_spacing, order=0, do_blur=False , as_type=np.int)
        
        img_space = (out_im_space- np.mean(out_im_space))/np.std(out_im_space)
        
        gt = Tensor(out_gt_space).permute(1,2,0).detach().numpy()
        img = Tensor(img_space).permute(1,2,0).detach().numpy()
        
        im_slices  = img.shape[2]-1
        gt_slices  = gt.shape[2]-1     # OBS: appical slices removed
        
        pad = 5                     # padding added to ensure correct cropping
    
        gt_p  = np.zeros((gt.shape[0]+pad,  gt.shape[1]+pad,  gt_slices))
        img_p = np.zeros((img.shape[0]+pad, img.shape[1]+pad, gt_slices))
        
        for j in range(0,gt_slices):
            img_p[:,:,j] = np.pad(img[:,:,j] ,((pad,0),(pad,0)), 'constant', constant_values=0)
            gt_p[:,:,j]  = np.pad(gt[:,:,j]  ,((pad,0),(pad,0)), 'constant', constant_values=0)
        
        c_slice   = int(np.floor(gt_slices/2))                           # Finding middle slice as it's assumed these will centered and never be empty
        bin_gt    = np.zeros((gt_p.shape[0],gt_p.shape[1]))              # Preallocate
        bin_gt[gt_p[:,:,c_slice] >= 1] = 1                               # Binarize annotations
        center[i,0],center[i,1]   = scipy.ndimage.center_of_mass(bin_gt) # Find center of gravity of annotated cardiac structure
        
        cropped_gt = np.zeros((H,W,gt_slices))                           # Preallocate
        cropped_im = np.zeros((H,W,gt_slices))                           # Preallocate
                
        
        for j in range(0,im_slices):
            top   = int(np.ceil(center[i,0] - (128/2)))
            bot   = int(np.ceil(center[i,0] + (128/2)))
            
            left  = int(np.ceil(center[i,1] - (128/2)))
            right = int(np.ceil(center[i,1] + (128/2)))
            
            cropped_gt[:,:,j] = gt_p[top:bot,left:right,j]
            cropped_im[:,:,j] = img_p[top:bot,left:right,j]                  
        
        in_image = np.expand_dims(cropped_im,0)
        in_image = Tensor(in_image).permute(3,0,1,2).detach().numpy()
        im_crop.append(in_image)
        
        in_gt = Tensor(cropped_gt).permute(2,0,1).detach().numpy()
        gt_crop.append(in_gt)
        
    
    return im_crop, gt_crop




















