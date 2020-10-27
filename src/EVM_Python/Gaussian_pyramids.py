import numpy as np
import cv2


# Build Gaussian Pyramid from Image (already cv2.imread -ed)
def build_Gpyr(frame, levels):
    G = np.ndarray(shape=frame.shape, dtype=np.float)
    G[:] = frame
    Gpyr = [G]

    for i in range(levels):
        G = cv2.pyrDown(G)
        Gpyr.append(G)

    return Gpyr
