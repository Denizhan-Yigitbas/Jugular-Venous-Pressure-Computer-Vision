import numpy as np


def bgr2yiq(vid_bgr):

    bgr2yiq_matrix = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]])
    # Wrong since frames are stored in BGR in OpenCV

    # BGR to YIQ conversion matrix (OpenCV BGR storing accounted for)
    #bgr2yiq_matrix = np.array([np.flip([0.299, 0.587, 0.114]), np.flip([0.596, -0.274, -0.322]), np.flip([0.211, -0.523, 0.312])])

    # Conversion
    vid_yiq = np.dot(vid_bgr, bgr2yiq_matrix.T)

    return vid_yiq


def yiq2bgr(frame_yiq):

    yiq2bgr_matrix = np.array([[1, 0.956, 0.619], [1, -0.272, -0.647], [1, -1.106, 1.703]])

    #yiq2bgr_matrix = np.array([[1, -1.106, 1.703], [1, -0.272, -0.647], [1, 0.956, 0.619]])

    frame_bgr = np.dot(frame_yiq, yiq2bgr_matrix.T)
    return frame_bgr
