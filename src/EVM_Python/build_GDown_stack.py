# Generated with SMOP  0.41
from libsmop import *
import cv2
import numpy as np

# GDOWN_STACK = build_GDown_stack(VID_FILE, START_INDEX, END_INDEX, LEVEL)
# 
# Apply Gaussian pyramid decomposition on VID_FILE from START_INDEX to
# END_INDEX and select a specific band indicated by LEVEL
# 
# GDOWN_STACK: stack of one band of Gaussian pyramid of each frame 
# the first dimension is the time axis
# the second dimension is the y axis of the video
# the third dimension is the x axis of the video
# the forth dimension is the color channel
# 
# Copyright (c) 2011-2012 Massachusetts Institute of Technology, 
# Quanta Research Cambridge, Inc.

# Authors: Hao-yu Wu, Michael Rubinstein, Eugene Shih,
# License: Please refer to the LICENCE file
# Date: June 2012
from blurDnClr import blurDnClr


class Struct:
    pass


def build_GDown_stack(vidFile, startIndex, endIndex, level):
    # Input Video Properties
    vid = cv2.VideoCapture(vidFile)

    # if vid.isOpened():
    vidWidth = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    vidHeight = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    nChannels = 3

    temp = Struct()
    temp.cdata = np.zeros((vidHeight, vidWidth, nChannels), dtype=int)  # Is this right?
    temp.colormap = []

    # Read First Frame
    vid.set(1, startIndex)
    temp.cdata = vid.read()[1]

    # Write first frame to image and run Gaussian pyramid decomposition using Pyrdown from cv2

    gpf = [temp]
    for i in range(0, level):
        temp = cv2.pyrDown(temp)
        gpf.append(temp)

    blurred = gpf[-1]

    # Create Pyr Stack
    GDown_stack = np.zeros((endIndex - startIndex + 1, np.shape(blurred)[0], np.shape(blurred)[1], np.shape(blurred)[2]))
    GDown_stack[1, :, :, :] = blurred

    k = 1
    gpf = []

    # Fill in stack starting from 2nd frame to end frame
    for i in range(startIndex + 1, endIndex):
        k = k + 1

        vid.set(1, i)
        temp.cdata = vid.read()[1]

        gpf = [temp]
        for j in range(0, level):
            temp = cv2.pyrDown(temp)
            gpf.append(temp)

        blurred = gpf[-1]

        GDown_stack[k, :, :, :] = blurred

    return GDown_stack
