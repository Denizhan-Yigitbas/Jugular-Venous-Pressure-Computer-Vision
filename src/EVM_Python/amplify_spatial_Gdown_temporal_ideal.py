# Generated with SMOP  0.41
from libsmop import *
from pathlib import Path
import cv2
import numpy as np
from build_GDown_stack import build_GDown_stack
from ideal_bandpassing import ideal_bandpassing


# amplify_spatial_Gdown_temporal_ideal(vidFile, outDir, alpha,
#                                      level, fl, fh, samplingRate, 
#                                      chromAttenuation)

# Spatial Filtering: Gaussian blur and down sample
# Temporal Filtering: Ideal bandpass
# 
# Copyright (c) 2011-2012 Massachusetts Institute of Technology, 
# Quanta Research Cambridge, Inc.

# Authors: Hao-yu Wu, Michael Rubinstein, Eugene Shih,
# License: Please refer to the LICENCE file
# Date: June 2012


class Struct:
    pass


def amplify_spatial_Gdown_temporal_ideal(vidFile, outDir, alpha, level, fl, fh, samplingRate, chromAttenuation):

    # THis should belong in separate file since they all require this

    # Input and Output video names
    parts = vidFile.split('/')
    vidName = parts[len(parts) - 1].split('.')[0]

    dirname = outDir
    filename = vidName + '-ideal-from-' + str(fl) + '-to-' + str(fh) + '-alpha-' + str(alpha) + '-level-' + str(
        level) + '-chromAtn-' + str(chromAttenuation)
    suffix = '.avi'
    outName = Path(dirname, filename).with_suffix(suffix)

    # Input Video Properties
    vid = cv2.VideoCapture(vidFile)

    # if vid.isOpened():
    vidWidth = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    vidHeight = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    nChannels = 3
    fr = vid.get(cv2.CAP_PROP_FPS)
    length = vid.get(cv2.CAP_PROP_FRAME_COUNT)

    temp = Struct()
    temp.cdata = np.zeros((vidHeight, vidWidth, nChannels), dtype=int)  # Is this right?
    temp.colormap = []

    startIndex = 1
    endIndex = length - 10

    # Write Output Video (amplified)
    vidOut = cv2.VideoWriter(outName, fr)

    # vidOut.FrameRate = copy(fr)

    vidOut.open()  # needed?

    """
    vid = VideoReader(vidFile)
    vidHeight = vid.Height
    vidWidth = vid.Width
    nChannels = 3
    fr = vid.FrameRate
    len_ = vid.NumberOfFrames
    temp = struct('cdata', zeros(vidHeight, vidWidth, nChannels, 'uint8'), 'colormap', [])
    
    startIndex = 1
    endIndex = len_ - 10

    vidOut = VideoWriter(outName)
    vidOut.FrameRate = copy(fr)

    open_(vidOut)
    """

    # compute Gaussian blur stack
    print('Spatial filtering...')
    Gdown_stack = build_GDown_stack(vidFile, startIndex, endIndex, level)
    print('Finished')

    # Temporal filtering
    print('Temporal filtering...')
    filtered_stack = ideal_bandpassing(Gdown_stack, 1, fl, fh, samplingRate)
    print('Finished')

    # Amplify
    filtered_stack[:, :, :, 1] = filtered_stack[:, :, :, 1] * alpha
    filtered_stack[:, :, :, 2] = filtered_stack[:, :, :, 2] * alpha * chromAttenuation
    filtered_stack[:, :, :, 3] = filtered_stack[:, :, :, 3] * alpha * chromAttenuation

    # Render on the input video
    print('Rendering...')
    # output video
    k = 0
    for i in range(startIndex - 1, endIndex):  # recheck indexes for range

        k = k + 1  # Move to bottom

        vid.set(1, i)

        temp.cdata = vid.read()[1]  # read the ith frame of input video
        frame = temp

        filtered = np.squeeze(filtered_stack[k, :, :, :])

        filtered = cv2.resize(filtered, [vidHeight, vidWidth])

        filtered = filtered + frame

        filtered[filtered > 1] = 1
        filtered[filtered < 0] = 0

        vidOut.write(filtered)

    print('Finished')
    close_(vidOut)
    return
