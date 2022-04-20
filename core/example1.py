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
# RAW_NAME = 'Mito_72_s&tCrop_mod.tif'
# RAW_NAME = 'Mito_74_DUP_s&tCrop.tif'
RAW_NAME = 'C2-18-07-04_DC_67xYW(F1)_b7_KltReady.tif'

''' 2) Tracking options '''
TIME_CLIP = 5 # must be >=1

#%% Initialize

# Open data
raw = io.imread(ROOT_PATH + RAW_NAME) 

#%%

# Parameters for feature detection
feature_params = dict(
    maxCorners=3000,
    qualityLevel=0.001,
    minDistance=7,
    blockSize=7,
	useHarrisDetector=True)

# Parameters for optical flow
lk_params = dict(
    winSize = (31,31),
    maxLevel = 3,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

#%%

# Convert raw as uint8
raw = as_uint8(raw, int_range=0.999)



# Get image & features (t0)
img0 = raw[0,:,:]
f0 = cv2.goodFeaturesToTrack(
    img0, mask=None, **feature_params)


# Create empty variables
nt = raw.shape[0]
nf = f0.shape[0]
klt_data = []


features = np.zeros_like(raw, dtype='uint16')
tracks = np.zeros_like(raw, dtype='uint16')
for i in range(1,nt):
    
    # Get current image
    img1 = raw[i,:,:]
    
    # Calculate optical flow (between t0 and current)
    f1, st, err = cv2.calcOpticalFlowPyrLK(
        img0, img1, f0, None, **lk_params)
    
    # Select good features
    valid_f1 = f1#[st==1]
    valid_f0 = f0#[st==1]
    
    # Make a display
    for j,(new,old) in enumerate(zip(valid_f1,valid_f0)):
        
        a,b = new.ravel().astype('int')
        c,d = old.ravel().astype('int')
        tracks[i,:,:] = cv2.line(tracks[i,:,:], (a,b), (c,d), j, 1)
        features[i,:,:] = cv2.circle(features[i,:,:], (a,b), 1, j, 1)
        # tracks[i,:,:] = cv2.line(tracks[i,:,:], (a,b), (c,d), (255,255,255), 1)
        # features[i,:,:] = cv2.circle(features[i,:,:], (a,b), 1, (255,255,255), 1)
        
    # Update previous image & features 
    img0 = img1
    f0 = valid_f1.reshape(-1,1,2) 

    klt_data.append((f1.shape[0]))

#%%

io.imsave(ROOT_PATH+'/temp/'+RAW_NAME[0:-4]+'_tracks.tif', tracks, check_contrast=False) 
io.imsave(ROOT_PATH+'/temp/'+RAW_NAME[0:-4]+'_features.tif', features, check_contrast=False) 

