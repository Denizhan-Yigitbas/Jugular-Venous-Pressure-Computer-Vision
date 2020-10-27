import numpy as np
import cv2
from Gaussian_pyramids import build_Gpyr


# Build Laplacian Pyramid from Image (already cv2.imread -ed)
def build_Lpyr(frame, levels):
    Gpyr = build_Gpyr(frame, levels)
    Lpyr = [Gpyr[levels - 1]]

    for i in range(levels - 1, 0, -1):
        GE = cv2.pyrUp(Gpyr[i])
        L = cv2.subtract(Gpyr[i - 1], GE)
        Lpyr.append(L)

    return Lpyr


# Build Laplacian Pyramid with all frames of video
def video_Lpyr(vid, levels, start_index, end_index):

    # Initialize Laplacian pyramid of full input video (each row contains array of frames at pyramid level)
    vpyr = np.empty(shape=(levels, 1), dtype=float)

    for i in range(start_index, end_index):
        vid.set(1, i)
        frame = vid.read()[1]
        frame_Lpyr = build_Lpyr(frame, levels)

        for j in range(levels):
            vpyr[j].append(frame_Lpyr[j])

    return vpyr
