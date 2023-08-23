#%% Imports -------------------------------------------------------------------

import cv2
import numpy as np
from pystackreg import StackReg
from skimage.draw import line
from skimage.morphology import dilation, square

import time

#%% KLT -----------------------------------------------------------------------

#%% Execute -------------------------------------------------------------------

import napari
from skimage import io
from pathlib import Path

# Get name and open data
# stack_name = '18-07-11_40x_GBE_UtrCH_Ctrl_b1_Lite_uint8.tif'
stack_name = '230616_RPE1_cyclg_01_SIM².tif'
stack = io.imread(Path('data') / stack_name)

# -----------------------------------------------------------------------------

# Feature detection
feat_params = dict(
    maxCorners=1000,
    qualityLevel=0.0001,
    minDistance=5,
    blockSize=5,
	useHarrisDetector=True
    )

# Optical flow
flow_params = dict(
    winSize=(11, 11),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01)
    )

# -----------------------------------------------------------------------------

start = time.time()
print('KLT')

klt_data = {
    'xCoords': [],
    'yCoords': [],
    'dXY': [],
    'status': [],
    'errors': [],
    }

# Get frame & features (t0)
frm0 = stack[0,:,:]
f0 = cv2.goodFeaturesToTrack(
    frm0, mask=None, **feat_params
    )

for t in range(1, stack.shape[0]):
    
    # Get current image
    frm1 = stack[t,:,:]
    
    # Compute optical flow (between f0 and f1)
    f1, status, errors = cv2.calcOpticalFlowPyrLK(
        frm0, frm1, f0, None, **flow_params
        )
    
    # Format outputs
    errors = errors.squeeze().astype(float);
    status = status.squeeze().astype(float); 
    f0 = f0.squeeze(); f1 = f1.squeeze()
    f0[f0[:,0] >= frm0.shape[1]] = np.nan
    f0[f0[:,1] >= frm0.shape[0]] = np.nan
    f1[f1[:,0] >= frm1.shape[1]] = np.nan
    f1[f1[:,1] >= frm1.shape[0]] = np.nan
    f1[status == 0] = np.nan
    
    # Measure distances
    dX = f1[:,0] - f0[:,0]
    dY = f1[:,1] - f0[:,1]
    dXY = np.linalg.norm(f1 - f0, axis=1) 
        
    # Append klt_data
    if t == 1:
        nan = np.full_like(status, np.nan)
        klt_data['xCoords'].append(f0[:,0])
        klt_data['yCoords'].append(f0[:,1])
        klt_data['status'].append(nan)
        klt_data['errors'].append(nan)
        klt_data['dXY'].append(nan)
    klt_data['xCoords'].append(f1[:,0])
    klt_data['yCoords'].append(f1[:,1])
    klt_data['status'].append(status)
    klt_data['errors'].append(errors)
    klt_data['dXY'].append(dXY)
        
    # Update previous frame & features 
    frm0 = frm1
    f0 = f1.reshape(-1,1,2) 
 
end = time.time()
print(f'  {(end-start):5.3f} s')
 
# ----------------------------------------------------------------------------- 

start = time.time()
print('Display')

# Create empty diplay arrays
ftsRaw = np.zeros_like(stack, dtype=bool)
tksRaw = np.zeros_like(stack, dtype=bool)
ftsLab = np.zeros_like(stack, dtype='uint16')
# tksLab = np.zeros_like(stack, dtype='uint16')
ftsdXY = np.zeros_like(stack, dtype=float)
# tksdXY = np.zeros_like(stack, dtype=float)
ftsErr = np.zeros_like(stack, dtype=float)
# tksErr = np.zeros_like(stack, dtype=float)

for t in range(stack.shape[0]):

    # Extract variables   
    x1s = klt_data['xCoords'][t]
    y1s = klt_data['yCoords'][t]
    dXY = klt_data['dXY'][t]
    errors = klt_data['errors'][t]
    labels = np.arange(x1s.shape[0]) + 1
    
    # Remove non valid data
    valid_idx = ~np.isnan(x1s)
    x1s = x1s[valid_idx].astype(int)
    y1s = y1s[valid_idx].astype(int)
    dXY = dXY[valid_idx]
    errors = errors[valid_idx]
    labels = labels[valid_idx]
    
    # Fill features display arrays
    ftsRaw[t, y1s, x1s] = True
    ftsLab[t, y1s, x1s] = labels
    ftsdXY[t, y1s, x1s] = dXY
    ftsErr[t, y1s, x1s] = errors
    
    # Fill tracks display arrays
    if t > 0:
        x0s = klt_data['xCoords'][t-1]
        y0s = klt_data['yCoords'][t-1]
        x0s = x0s[valid_idx].astype(int)
        y0s = y0s[valid_idx].astype(int)
        for x0, y0, x1, y1 in zip(x0s, y0s, x1s, y1s):
            rr, cc = line(y0, x0, y1, x1)
            tksRaw[t,rr,cc] = True

    # Dilate display arrays
    dilSize = 3
    ftsRaw[t,...] = dilation(ftsRaw[t,...], footprint=square(dilSize))
    ftsLab[t,...] = dilation(ftsLab[t,...], footprint=square(dilSize))
    ftsdXY[t,...] = dilation(ftsdXY[t,...], footprint=square(dilSize))
    ftsErr[t,...] = dilation(ftsErr[t,...], footprint=square(dilSize))
    

end = time.time()
print(f'  {(end-start):5.3f} s')

# -----------------------------------------------------------------------------   

viewer = napari.Viewer()
viewer.add_image(stack)
# viewer.add_image(ftsRaw, blending='additive')
viewer.add_image(tksRaw, blending='additive')
viewer.add_labels(ftsLab, blending='additive')
# viewer.add_image(ftsdXY, blending='additive', colormap='turbo')
# viewer.add_image(ftsErr, blending='additive', colormap='turbo')

