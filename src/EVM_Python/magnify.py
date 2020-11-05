from pathlib import Path
import cv2
import numpy as np
from pyramids import build_Gpyr, build_Lpyr, video_Lpyr, video_Gpyr_onelayer
from OpenCV import video_name
from collapse import Lpyr_collapse
from filters import temporal_bandpass_filter
from colorspace import bgr2yiq, yiq2bgr
from video import input_video


def magnify(vidFile, outDir, alpha, level, fl, fh,
            chromAttenuation):  # samplingRate, chromAttenuation, pyramid_func, filter_type

    # Input Video Properties and Video array
    vidWidth, vidHeight, fr, length, vid_bgr = input_video(vidFile)

    # Convert RGB video to YIQ???
    vid_yiq = bgr2yiq(vid_bgr)

    """
    vidWidth = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    vidHeight = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nChannels = 3
    fr = vid.get(cv2.CAP_PROP_FPS)
    length = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    """
    # startIndex = 0
    endIndex = int(length) - 1

    # Build the full input video (pyramid_func) pyramid
    print('Spatial filtering...')
    video_pyramid = video_Lpyr(vid_yiq, level)

    print('Finished')

    # Temporal filtering and Amplification
    print('Temporal filtering and Amplifying...')

    # Laplacian Pyramid

    for i in range(2, len(video_pyramid)):      # Ignore top and bottom layer due to noise and etc?

        # Pyramid levels as spatial band
        spatial_band = video_pyramid[i]

        # Temporal Filtering
        filtered = temporal_bandpass_filter(spatial_band, fr, fl, fh)

        # Amplification
        filtered *= alpha

        # Chromatic Attenuation
        filtered[:][:][:][1] *= chromAttenuation
        filtered[:][:][:][2] *= chromAttenuation

        # Add amplified signal to original
        video_pyramid[i] += filtered

    print('Finished')

    # Collapse Magnified video pyramid and write to output video
    print('Collapsing pyramid and writing video...')

    # Write Output Video (amplified)
    # outName = video_name(vidFile, outDir, alpha, level, fl, fh)
    outName = "/Users/sang-hyunlee/Desktop/output4.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # account for any video format
    vidOut = cv2.VideoWriter(outName, fourcc, fr, (vidWidth, vidHeight))

    # Laplacian Pyramid

    c = 0

    while c <= endIndex:
        # Extract pyramid of amplified frame from video pyramid
        amplified_frame_pyr = [video_pyramid[x][c] for x in range(len(video_pyramid))]

        # Collapse the amplified frame pyramid
        amplified_frame_yiq = Lpyr_collapse(amplified_frame_pyr)

        # Convert to uint8 format (normalize by data type, rather than max value within array)
        #info = np.finfo(amplified_frame.dtype)
        #amplified_frame = np.divide(amplified_frame, info.max)
        #amplified_frame = np.multiply(amplified_frame, 255)

        amplified_frame_bgr = yiq2bgr(amplified_frame_yiq)

        # Remove unwanted values prior to int conversion
        amplified_frame_bgr[amplified_frame_bgr > 255] = 255
        amplified_frame_bgr[amplified_frame_bgr < 0] = 0

        # Conversion to uint8
        final_frame = amplified_frame_bgr.astype(np.uint8)   # Might be causing issues!!!! Maybe remove <0 and set >255 to 255

        # Write to Output video
        vidOut.write(final_frame)

        c += 1

    # One layer (highest) of Gaussian amplified
    """
    filtered1 = temporal_bandpass_filter(video_pyramid1, fr, fl, fh)
    filtered1 *= alpha
    filtered1[:][:][:][1] *= chromAttenuation
    filtered1[:][:][:][2] *= chromAttenuation

    filtered2 = temporal_bandpass_filter(video_pyramid2, fr, fl, fh)
    filtered2 *= alpha
    filtered2[:][:][:][1] *= chromAttenuation
    filtered2[:][:][:][2] *= chromAttenuation

    filtered3 = temporal_bandpass_filter(video_pyramid3, fr, fl, fh)
    filtered3 *= alpha
    filtered3[:][:][:][1] *= chromAttenuation
    filtered3[:][:][:][2] *= chromAttenuation

    filtered4 = temporal_bandpass_filter(video_pyramid4, fr, fl, fh)
    filtered4 *= alpha
    filtered4[:][:][:][1] *= chromAttenuation
    filtered4[:][:][:][2] *= chromAttenuation

    for i in range(len(vid_yiq)):
        filtered_resized1 = cv2.resize(filtered1[i], (vidWidth, vidHeight))
        filtered_resized2 = cv2.resize(filtered2[i], (vidWidth, vidHeight))
        filtered_resized3 = cv2.resize(filtered3[i], (vidWidth, vidHeight))
        filtered_resized4 = cv2.resize(filtered4[i], (vidWidth, vidHeight))

        frame = vid_yiq[i] + filtered_resized1 + filtered_resized2 + filtered_resized3 + filtered_resized4
        amplified_frame_bgr = yiq2bgr(frame)

        # Remove unwanted values prior to int conversion
        amplified_frame_bgr[amplified_frame_bgr > 255] = 255
        amplified_frame_bgr[amplified_frame_bgr < 0] = 0

        # Conversion to uint8
        final_frame = amplified_frame_bgr.astype(np.uint8)  # Might be causing issues!!!! Maybe remove <0 and set >255 to 255

        # Write to Output video
        vidOut.write(final_frame)
    """
    print('Finished')

    # Release everything at the end
    vidOut.release()

    #vidWidth2, vidHeight2, fr2, length2, vid_bgr2 = input_video("/Users/sang-hyunlee/Desktop/output4.mp4")
    #a = vid_bgr2[0]
    #b = vid_bgr2[10]

    cv2.destroyAllWindows()