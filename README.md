# Speciale2021
Master Thesis 2021

Bayes U-Net 
- UNET_GPU_phase.py is used for training UNet using GPU
- RUN_UNET_analysis.py is used for testing UNET using CPU 
  - Dropout: 0.5
  - Learning rate: 0.0001
  - Loss: CrossEntropy
  - Training and eval
  
OBS: Patients in 5 subgroups: 
- normal cardiac function 
- four disease groups: dilated cardiomyopathy, hypertrophic cardiomyopathy, heart failure with infarction, and right ventricular abnormality.  

Detection Network 
- sResNet (not working yet) 
- 
Inspired by https://github.com/toologicbv/cardiacSegUncertainty/tree/b2ee4f0458b746fa343127853ad9a56caa4bdcec
Data from: https://acdc.creatis.insa-lyon.fr/

