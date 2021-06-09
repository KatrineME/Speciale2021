#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 10:46:28 2021

@author: michalablicher
"""




#%% Run model

#%% Load Model
#PATH_model = "C:/Users/katrine/Documents/GitHub/Speciale2021/trained_Unet_testtest.pt"
#PATH_state = "C:/Users/katrine/Documents/GitHub/Speciale2021/trained_Unet_testtestate.pt"

#PATH_model_es = '/Users/michalablicher/Desktop/Trained_Unet_CE_sys_sub_batch_100.pt'
PATH_model_ed = '/Users/michalablicher/Desktop/Trained_Detection_dia_state.pt'

#%% Load model
#unet_es = torch.load(PATH_model_es, map_location=torch.device('cpu'))
unet_es = torch.load(PATH_model_ed, map_location=torch.device('cpu'))

unet_es.eval()
out_trained_es = unet_es(Tensor(im_test_es_res))
out_image_es   = out_trained_es["softmax"]

#im_flat_test_ed = im_flat_test_ed.cuda()

#unet_ed.eval()
#out_trained_ed = unet_ed(Tensor(im_test_ed_sub))
#out_image_ed   = out_trained_ed["softmax"]

#%% One hot encoding

seg_met_dia = np.argmax(out_image_ed.detach().cpu().numpy(), axis=1)

seg_dia = torch.nn.functional.one_hot(torch.as_tensor(seg_met_dia), num_classes=4).detach().cpu().numpy()
ref_dia = torch.nn.functional.one_hot(Tensor(gt_test_ed_sub).to(torch.int64), num_classes=4).detach().cpu().numpy()

seg_met_sys = np.argmax(out_image_es.detach().cpu().numpy(), axis=1)

seg_sys = torch.nn.functional.one_hot(torch.as_tensor(seg_met_sys), num_classes=4).detach().cpu().numpy()
ref_sys = torch.nn.functional.one_hot(Tensor(gt_test_es_res).to(torch.int64), num_classes=4).detach().cpu().numpy()


#%% E-map
import scipy.stats

#emap = np.zeros((out_image_ed.shape[0],out_image_ed.shape[2],out_image_ed.shape[3]))
emap = np.zeros((out_image_es.shape[0],out_image_es.shape[2],out_image_es.shape[3]))

for i in range(0, emap.shape[0]):

    out_img = (out_image_es[i,:,:].detach().cpu().numpy())
    entropy2 = scipy.stats.entropy(out_img)
    
    # Normalize 
    m_entropy   = np.max(entropy2)
    entropy     = entropy2/m_entropy
    emap[i,:,:] = entropy

emap = np.expand_dims(emap, axis=1)

#%% Plot
#% Wrap all inputs together
im     = Tensor(im_test_es_res)
umap   = Tensor(emap)
seg    = Tensor(np.expand_dims(seg_met_sys, axis=1))

image = 2

plt.figure(dpi=200)
plt.subplot(1,4,1)
plt.subplots_adjust(wspace = 0.4)
plt.imshow(im[image,0,:,:])
plt.title('cMRI') 
plt.subplot(1,4,2)
plt.imshow(seg[image,0,:,:])
plt.title('Segmentation') 
plt.subplot(1,4,3)
plt.imshow(umap[image,0,:,:])   
plt.title('U-map') 
plt.subplot(1,4,4)
plt.imshow(gt_test_es_res[image,:,:])   
plt.title('ref') 

input_concat = torch.cat((im,umap,seg), dim=1)


out_test    = model(input_concat)
output_test = out_test['softmax'].detach().numpy()

#%% Visualize output from detection network
image = 2

k = np.zeros((output_test.shape[0],2,16,16))

for i in range (0,output_test.shape[0]):
    k[i,:,:,:] = output_test[i,:,:,:] > 0.1
    
plt.figure(dpi=200)
plt.subplot(1,3,1)
plt.imshow(output_test[image,0,:,:])
plt.title('no seg. failure')
plt.colorbar(fraction=0.05)
plt.subplots_adjust(hspace = 0.05, wspace = 0.5)

plt.subplot(1,3,2)
plt.imshow(output_test[image,1,:,:])
plt.title('seg. failure')
plt.colorbar(fraction=0.05)

plt.subplot(1,3,3)
plt.imshow(k[image,1,:,:])
plt.title('bin 0.1')
#plt.colorbar(fraction=0.05)
#%%
#% Upsample
image = 2
upper_image = image - 1
lower_image = image + 1

#test_im = Tensor(np.expand_dims(output_test[lower_image:upper_image,1,:,:],axis=0))
#test_im = Tensor(np.expand_dims(output_test[upper_image:lower_image,1,:,:],axis=0))
test_im = Tensor(np.expand_dims(k[upper_image:lower_image,1,:,:],axis=0))

up = nn.Upsample((128,128), mode='bilinear', align_corners=True)

up_im = up(test_im) > 0

#up_im[up_im > 0] = 1

plt.figure(dpi=200)
plt.subplot(1,3,1)
plt.subplots_adjust(wspace = 0.4)
plt.imshow(input_concat[image,2,:,:])
#plt.imshow(up_im[0,0,:,:])
plt.imshow(input_concat[image,0,:,:], alpha= 0.4)
plt.title('Segmentation')
plt.subplot(1,3,2)
plt.imshow(up_im[0,1,:,:])
plt.imshow(input_concat[image,0,:,:], alpha= 0.4)
plt.title('Error patch')
plt.subplot(1,3,3)
plt.imshow(up_im[0,1,:,:])
plt.imshow(np.argmax((ref_sys[image,:,:,:]),axis=2), alpha= 0.6)
plt.imshow(input_concat[image,0,:,:], alpha= 0.4)
plt.title('Reference w. error')


#%%
k = np.zeros((output_test.shape[0],2,16,16))
test_im = np.zeros((output_test.shape[0],2,16,16))
up_im = np.zeros((output_test.shape[0],2,128,128))

up = nn.Upsample((128,128), mode='bilinear', align_corners=True)

for i in range (1,output_test.shape[0]):
    k[i,:,:,:] = output_test[i,:,:,:] > 0.1
    upper_image = i - 1
    lower_image = i + 1
    test_im = Tensor(np.expand_dims(k[upper_image:lower_image,1,:,:],axis=0))

    up_im[i,:,:,:] = up(test_im) > 0


#%%
plt.figure(dpi=200)
for i in range(0,36):
    plt.subplot(8,6,i+1)
    plt.subplots_adjust(wspace = 0.2)
    plt.imshow(up_im[i,1,:,:])
    plt.imshow(input_concat[i,0,:,:], alpha= 0.4)
    plt.imshow(input_concat[i,2,:,:], alpha= 0.4)
    plt.xticks(fontsize = 6)
    plt.yticks(fontsize = 6)


"""
