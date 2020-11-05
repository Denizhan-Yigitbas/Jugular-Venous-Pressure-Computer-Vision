import cv2


# Collapse a given Laplacian pyramid into a singular frame
def Lpyr_collapse(frame_pyramid):

    #frame = frame_pyramid[0]
    frame = frame_pyramid[-1]
    #for i in range(1, len(frame_pyramid)):  # for all levels
    for i in range(len(frame_pyramid) - 1, 0, -1):
        frame = cv2.pyrUp(frame)
        #frame = cv2.add(frame, frame_pyramid[i])
        frame = cv2.add(frame, frame_pyramid[i - 1])

    return frame

