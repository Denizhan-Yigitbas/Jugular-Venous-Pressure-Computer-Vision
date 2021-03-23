
import cv2
import numpy as np


def crop_video(video, min_x, min_y, max_x, max_y):
    """
    Opens the provided video (cv2.VideoCapture object) and extracts the frame data
    into a numpy array, cropping each frame at the provided coordinates.
    """
    # Some characteristics from the original video
    w_frame = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_frame = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = int(video.get(cv2.CAP_PROP_FPS))
    frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    h, w = max_y - min_y, max_x - min_x

    if w_frame <= w:
        min_x, max_x = 0, w_frame - 1
        w = w_frame

    if h_frame <= h:
        min_y, max_y = 0, h_frame - 1
        h = h_frame

    # Initialize frame data array
    video_stack = np.empty((frames, h, w, 3))

    for x in range(frames):
        ret, frame = video.read()

        # Save the cropped frame
        video_stack[x] = frame[min_y:max_y, min_x:max_x]

        if not ret:
            break

    return video_stack, fps, w, h
