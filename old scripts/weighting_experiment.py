# -*- coding: utf-8 -*-
"""
Created on Thu May 27 13:58:24 2021

@author: katrine
"""
import numpy   as np

image = np.ones((3,128,128))
image[0,:,:]*0
image[1,:,:]*0

s = np.log(1-image+1e-6)

ss = -1*np.sum(s,(1,2))

sss = np.sum(ss)

ssss = sss*32

print(ssss)