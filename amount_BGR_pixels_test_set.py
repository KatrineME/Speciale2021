# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 14:40:38 2021

@author: katrine
"""
ref_dia = torch.nn.functional.one_hot(Tensor(gt_test_ed_sub).to(torch.int64), num_classes=4)

s = torch.sum(ref_dia,axis=(1,2))

d = ((s/(128*128))*100).detach().numpy()

bgr = np.mean(d[:,0])