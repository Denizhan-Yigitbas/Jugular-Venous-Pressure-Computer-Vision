import cv2
import numpy as np
import skvideo.io
import skvideo.datasets

from pathlib import Path


# Outputs output video file name and directory
def video_name(vidFile, outDir, alpha, level, fl, fh):

    parts = vidFile.split('/')
    vidName = parts[len(parts) - 1].split('.')[0]

    dirname = outDir
    filename = vidName + '-ideal-from-' + str(fl) + '-to-' + str(fh) + '-alpha-' + str(alpha) + '-level-' + str(
        level) + '-chromAtn-'
    suffix = '.mp4'
    outName = dirname + filename + suffix

    return outName


def input_video(vidFile):

    vid = cv2.VideoCapture(vidFile)

    # Input Video Properties
    vidWidth = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    vidHeight = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fr = vid.get(cv2.CAP_PROP_FPS)
    length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    # Build array of frames in chronological order ??
    vid_array = np.ndarray(shape=(length, vidHeight, vidWidth, 3), dtype=np.float)

    for i in range(0, length):
        vid.set(cv2.CAP_PROP_POS_MSEC, i)
        ret, frame = vid.read()
        if ret:
            vid_array[i] = frame
        else:
            continue

    vid.release()

    return vidWidth, vidHeight, fr, length, vid_array

