#%%

import time
from skimage import io
from matplotlib import pyplot as plt

import cv2
import numpy as np

#%%

from tools.dtype import as_uint8

#%% Parameters

''' 1) Get paths '''

ROOT_PATH = '../data/'
# RAW_NAME = 'Mito_72_s&tCrop.tif'
RAW_NAME = 'Mito_74_DUP_s&tCrop.tif'

#%% Initialize

# Open data
raw = io.imread(ROOT_PATH + RAW_NAME) 

#%%

# Parameters for feature detection
feature_params = dict(
    maxCorners=200,
    qualityLevel=0.1,
    minDistance=7,
    blockSize=7,
	useHarrisDetector=False)

# Parameters for optical flow
lk_params = dict(
    winSize  = (21,21),
    maxLevel = 3,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

#%%

img = as_uint8(raw[0,:,:], int_range=0.999)
corners = cv2.goodFeaturesToTrack(img, mask=None, **feature_params)
idx = (corners[:,0,1].astype('int'), corners[:,0,0].astype('int'))
display = np.zeros_like(img)
display[idx] = 1

#%%

# Convert raw as uint8
raw = as_uint8(raw, int_range=0.999)

mask = np.zeros_like(raw)
# Run KLT tracking
for i in range(1,raw.shape[0]):
    
    # Get features (at t0)
    f0 = cv2.goodFeaturesToTrack(raw[i-1,:,:], mask=None, **feature_params)
    
    # Calculate optical flow (between t0 and t+1)
    f1, st, err = cv2.calcOpticalFlowPyrLK(raw[i-1,:,:], raw[i,:,:], f0, None, **lk_params)
    
    # Select good features
    valid_f1 = f1[st==1]
    valid_f0 = f0[st==1]
    
    temp_mask = np.zeros_like(raw[i-1,:,:])
    color = np.random.randint(0,255,(100,3))    
    for j,(new,old) in enumerate(zip(valid_f1,valid_f0)):
        
        a,b = new.ravel().astype('int')
        c,d = old.ravel().astype('int')
        mask[i,:,:] = cv2.line(mask[i,:,:], (a,b), (c,d), (255,255,255), 1)
        # mask[i,:,:] = cv2.circle(mask[i,:,:], (a,b), 2, (255,255,255), 1)
        

#%%

io.imsave(ROOT_PATH+'/temp/'+RAW_NAME[0:-4]+'_display.tif', display.astype('uint8')*255, check_contrast=False) 
io.imsave(ROOT_PATH+'/temp/'+RAW_NAME[0:-4]+'_mask.tif', mask.astype('uint8')*255, check_contrast=False) 
io.imsave(ROOT_PATH+'/temp/'+RAW_NAME[0:-4]+'_img.tif', img, check_contrast=False) 


