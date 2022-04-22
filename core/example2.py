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
RAW_NAME = 'Mito_74_DUP_s&tCrop_mod.tif'
# RAW_NAME = 'C2-18-07-04_DC_67xYW(F1)_b7_KltReady.tif'
# RAW_NAME = '18-07-11_40x_GBE_UtrCH_Ctrl_b1_Lite_uint8.tif'

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
    minDistance=5,
    blockSize=5,
	useHarrisDetector=True)

# Parameters for optical flow
lk_params = dict(
    winSize = (5,5),
    maxLevel = 3,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

#%%

# # Convert raw as uint8
# raw = as_uint8(raw, int_range=0.999)

# # Get image & features (t0)
# img0 = raw[0,:,:]
# f0 = cv2.goodFeaturesToTrack(
#     img0, mask=None, **feature_params)

# # Create empty variables
# nt = raw.shape[0]
# nf = f0.shape[0]
# track_x = np.zeros((nt, nf))
# track_y = np.zeros((nt, nf))
# track_st = np.zeros((nt, nf))
# track_err = np.zeros((nt, nf))
# track_x[0,:] = np.squeeze(f0[:,:,0])
# track_y[0,:] = np.squeeze(f0[:,:,1])

# features = np.zeros_like(raw, dtype='uint16')
# tracks = np.zeros_like(raw, dtype='uint16')
# for t in range(1,nt):
    
#     # Get current image
#     img1 = raw[t,:,:]
    
#     # Calculate optical flow (between t0 and current)
#     f1, st, err = cv2.calcOpticalFlowPyrLK(
#         img0, img1, f0, None, **lk_params)
    
#     # # Select good features
#     # valid_f1 = f1#[st==1]
#     # valid_f0 = f0#[st==1]
    
#     # Make a display
#     for f, (temp_f1,temp_f0,temp_st) in enumerate(zip(f0,f1,st)):
        
#         x_f0, y_f0 = temp_f0.ravel().astype('int')
#         x_f1, y_f1 = temp_f1.ravel().astype('int')
#         temp_st = int(temp_st[0])
                  
#         if temp_st == 1:
#             features[t,:,:] = cv2.circle(
#                 features[t,:,:], (x_f1,y_f1), 1, 65535, 1)
#             tracks[t,:,:] = cv2.line(
#                 tracks[t,:,:], (x_f1,y_f1), (x_f0,y_f0), 65535, 1)
#         else:
#             # print(x_f1, y_f1)
#             features[t,:,:] = cv2.circle(
#                 features[t,:,:], (x_f1,y_f1), 1, 0, 1)
#             tracks[t,:,:] = cv2.line(
#                 tracks[t,:,:], (x_f1,y_f1), (x_f0,y_f0), 0, 1)
             
#     # Append tracking data
#     track_x[t,:] = np.squeeze(f1[:,:,0])
#     track_y[t,:] = np.squeeze(f1[:,:,1])
#     track_st[t,:] = np.squeeze(st)
#     track_err[t,:] = np.squeeze(err)
     
#     # Update previous image & features 
#     img0 = img1
#     f0 = f1.reshape(-1,1,2) 

#%%

start = time.time()
print('get tracks')

# Convert raw as uint8
raw = as_uint8(raw, int_range=0.999)

# Get image & features (t0)
img0 = raw[0,:,:]
f0 = cv2.goodFeaturesToTrack(
    img0, mask=None, **feature_params)

# Create variables
nt = raw.shape[0]
nf = f0.shape[0]
track_x = np.zeros((nt, nf))
track_y = np.zeros((nt, nf))
track_st = np.zeros((nt, nf))
track_err = np.zeros((nt, nf))
track_x[0,:] = np.squeeze(f0[:,:,0])
track_y[0,:] = np.squeeze(f0[:,:,1])

for t in range(1,nt):

    # Get current image
    img1 = raw[t,:,:]
    
    # Calculate optical flow (between t0 and current)
    f1, st, err = cv2.calcOpticalFlowPyrLK(
        img0, img1, f0, None, **lk_params)    
    
    # Append tracking data
    track_x[t,:] = np.squeeze(f1[:,:,0])
    track_y[t,:] = np.squeeze(f1[:,:,1])
    track_st[t,:] = np.squeeze(st)
    track_err[t,:] = np.squeeze(err)
    
    # Update previous image & features 
    img0 = img1
    f0 = f1.reshape(-1,1,2) 
    
end = time.time()
print(f'  {(end - start):5.3f} s')     
    
#%%    

start = time.time()
print('get dist')

def get_dist(track_x, track_y):
    
    coords = np.column_stack((
        np.reshape(track_x, (-1,1), order=('F')), 
        np.reshape(track_y, (-1,1), order=('F'))
        ))
        
    dist = np.linalg.norm(
        coords - np.roll(coords, -1, axis=0), 
        axis=1
        )
    
    dist = np.reshape(dist, track_x.shape, order=('F'))
    dist = np.roll(dist, 1, axis=0)
    dist[0,:] = 0
    
    return dist

dist = get_dist(track_x, track_y)

end = time.time()
print(f'  {(end - start):5.3f} s') 

import matplotlib.pyplot as plt
plt.hist(dist.ravel(),bins=100, range=(0,10))
plt.show()

#%%

# for f in range(nf):
    
#     tmp_dist = dist[:,f]


tmp_dist = dist[:,16]
tmp_idx = np.where(tmp_dist > 5)
    
    

#%%
start = time.time()
print('get display')

features = np.zeros_like(raw, dtype='float32')
tracks = np.zeros_like(raw, dtype='float32')
for t in range(1,nt):
    for f in range(nf):
        
        x0, y0 = int(track_x[t-1,f]), int(track_y[t-1,f])
        x1, y1 = int(track_x[t,f]), int(track_y[t,f])
               
        features[t,:,:] = cv2.circle(
            features[t,:,:], (x1,y1), 1, dist[t,f], 1)
        
        tracks[t,:,:] = cv2.line(
            tracks[t,:,:], (x1,y1), (x0,y0), dist[t,f], 1)
       
end = time.time()
print(f'  {(end - start):5.3f} s')         

#%%

io.imsave(ROOT_PATH+'/temp/'+RAW_NAME[0:-4]+'_tracks.tif', tracks, check_contrast=False) 
io.imsave(ROOT_PATH+'/temp/'+RAW_NAME[0:-4]+'_features.tif', features, check_contrast=False) 

