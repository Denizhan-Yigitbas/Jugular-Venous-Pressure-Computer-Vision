import numpy as np
import cv2


# Build Gaussian Pyramid from Image (already cv2.imread -ed)
def build_Gpyr(frame, levels):
    G = np.ndarray(shape=frame.shape, dtype=np.float)  # change later?
    G[:] = frame
    Gpyr = [G]
    for i in range(1, levels):  # level: 0 ~ levels -1 (total: levels)
        G = cv2.pyrDown(G)
        Gpyr.append(G)

    return Gpyr


# Build Laplacian Pyramid from Image (already cv2.imread -ed)
def build_Lpyr(frame, levels):
    # Initialize Gaussian pyramid of frame
    Gpyr = build_Gpyr(frame, levels)

    # Careful with levels. If the level size is odd, and the previous was odd, then pyrup won't work since it doubles (error with subtraction of different sizes)

    Lpyr = []

    # Build Laplacian pyramid from Gaussian
    for i in range(levels - 1):  # one less level than Gaussian since we take differences
        # Difference between level and expanded version of upper level in Gaussian
        GE = cv2.pyrUp(Gpyr[i + 1])
        L = cv2.subtract(Gpyr[i], GE)
        Lpyr.append(L)

    Lpyr.append(Gpyr[-1])

    """
    Lpyr = [Gpyr[levels - 1]]
    # Build Laplacian pyramid from Gaussian (my version)
    for i in range(levels - 1, 0, -1):  # one less level than Gaussian since we take differences
        # Difference between level and expanded version of upper level in Gaussian
        GE = cv2.pyrUp(Gpyr[i])
        L = cv2.subtract(Gpyr[i - 1], GE)
        Lpyr.append(L)

    """
    return Lpyr


# Build Laplacian Pyramid with all frames of video
def video_Lpyr(vid, levels):
    # Initialize Laplacian pyramid of full input video (each element contains array of frames at pyramid level)
    vpyr = [[] for x in range(levels)]

    # For each frame
    for i in range(len(vid)):
        frame = vid[i]
        frame_Lpyr = build_Lpyr(frame, levels)

        for j in range(levels):
            vpyr[j].append(frame_Lpyr[j])

    return vpyr


def video_Gpyr_onelayer(vid, levels, c):

    vpyr = [[] for x in range(len(vid))]

    for i in range(len(vid)):
        frame = vid[i]
        pyr = build_Gpyr(frame, levels)
        gaussian_frame = pyr[c]  # use only highest gaussian level             # initialize one time

        vpyr[i] = gaussian_frame

    return vpyr


def video_Gpyr(vid, levels):

    vpyr = [[] for x in range(len(vid))]

    for i in range(len(vid)):
        frame = vid[i]
        frame_Gpyr = build_Gpyr(frame, levels)

        for j in range(levels):

            vpyr[j].append(frame_Gpyr[j])

    return vpyr

