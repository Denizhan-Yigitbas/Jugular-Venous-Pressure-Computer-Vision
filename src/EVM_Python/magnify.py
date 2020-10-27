# Generated with SMOP  0.41
from pathlib import Path
import cv2
import numpy as np
from pyramids import build_Gpyr, build_Lpyr, video_Lpyr


class Struct:
    pass


def magnify(vidFile, outDir, alpha, level, fl, fh, samplingRate, chromAttenuation, pyramid_func, filter_type):
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

    startIndex = 0
    endIndex = length - 1

    # Write Output Video (amplified)
    vidOut = cv2.VideoWriter(outName, fr)

    # vidOut.FrameRate = copy(fr)

    vidOut.open()  # needed?

    # Build the full input video (pyramid_func) pyramid
    print('Spatial filtering...')
    video_pyramid = video_Lpyr(vid, level, startIndex, endIndex)
    print('Finished')

    # Temporal filtering and Amplification
    print('Temporal filtering and Amplifying...')
    for i in range(video_pyramid.shape[0]):
        # Pyramid levels as spatial band
        spatial_band = video_pyramid[i]

        # Temporal Filtering
        filtered = filter_type(spatial_band, 1, fl, fh, samplingRate)

        # Amplification
        filtered = ~

        # Add amplified signal to original
        video_pyramid[i] = video_pyramid[i] + filtered
    print('Finished')

    # Collapse Magnified video pyramid


    #


    # Amplify
    filtered_stack[:, :, :, 1] = filtered_stack[:, :, :, 1] * alpha
    filtered_stack[:, :, :, 2] = filtered_stack[:, :, :, 2] * alpha * chromAttenuation
    filtered_stack[:, :, :, 3] = filtered_stack[:, :, :, 3] * alpha * chromAttenuation

    # Render on the input video
    print('Rendering...')
    # output video


    print('Finished')
    close_(vidOut)
    return
