#%% Imports -------------------------------------------------------------------

import cv2
import numpy as np
from pystackreg import StackReg
from scipy.spatial import distance
from skimage.draw import circle_perimeter, line

import time

#%% KLT -----------------------------------------------------------------------

#%% Execute -------------------------------------------------------------------

import napari
from skimage import io
from pathlib import Path

from skimage.morphology import binary_dilation, disk

# Get name and open data
stack_name = '18-07-11_40x_GBE_UtrCH_Ctrl_b1_Lite_uint8.tif'
stack = io.imread(Path('data') / stack_name)

# -----------------------------------------------------------------------------

# Feature detection
feat_params = dict(
    maxCorners=300,
    qualityLevel=0.001,
    minDistance=5,
    blockSize=5,
	useHarrisDetector=True
    )

# Optical flow
flow_params = dict(
    winSize=(21,21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

# -----------------------------------------------------------------------------

klt_data = []

# Get frame & features (t0)
frm0 = stack[0,:,:]
f0 = cv2.goodFeaturesToTrack(
    frm0, mask=None, **feat_params
    )

for t in range(1, stack.shape[0]):
    
    # Get current image
    frm1 = stack[t,:,:]
    
    # Compute optical flow (between f0 and f1)
    f1, status, error = cv2.calcOpticalFlowPyrLK(
        frm0, frm1, f0, None, **flow_params
        )
    
    # Format outputs
    error = error.squeeze();
    status = status.squeeze(); 
    f0 = f0.squeeze(); f1 = f1.squeeze()
    f0[f0[:,0] >= frm0.shape[1]] = np.nan
    f0[f0[:,1] >= frm0.shape[0]] = np.nan
    f1[f1[:,0] >= frm1.shape[1]] = np.nan
    f1[f1[:,1] >= frm1.shape[0]] = np.nan
    f1[status == 0] = np.nan
        
    # Append klt_data
    if t == 1:
        klt_data.append((f0[:,0], f0[:,1]))
    klt_data.append((f1[:,0], f1[:,1], status, error))
        
    # Update previous frame & features 
    frm0 = frm1
    f0 = f1.reshape(-1,1,2) 
 
# ----------------------------------------------------------------------------- 

features = np.zeros_like(stack, dtype=bool)
for t in range(0, stack.shape[0]):
    
    xCoords = klt_data[t][0]
    yCoords = klt_data[t][1]
    xCoords = xCoords[~np.isnan(xCoords)].astype(int)
    yCoords = yCoords[~np.isnan(yCoords)].astype(int)
    features[t, yCoords, xCoords] = True
    features[t,...] = binary_dilation(features[t,...], footprint=disk(3))

# featID = np.zeros_like(stack, dtype='uint16')
# dist = np.linalg.norm(f1 - f0, axis=1)

# -----------------------------------------------------------------------------   

viewer = napari.Viewer()
viewer.add_image(stack)
viewer.add_image(features, blending='additive')
