import cv2
import numpy as np

file = '/Users/sang-hyunlee/Desktop/RotatedCirclescopy.jpeg'


def sticker_detection_coords(video_stack):
    """
        For an input frame of video, detects red stickers (circle/oval), finds the center x,y coordinates of each sticker in each
        frame, and returns the min_x, min_y, max_x, max_y.
    """

    # Set our filtering parameters
    # Initialize parameter setting using cv2.SimpleBlobDetector
    params = cv2.SimpleBlobDetector_Params()

    # Set Area filtering parameters
    params.filterByArea = False
    params.maxArea = 1000

    # Set Circularity filtering parameters
    params.filterByCircularity = False
    params.minCircularity = 0.3

    # Set Convexity filtering parameters
    params.filterByConvexity = True
    params.minConvexity = 0.2

    # Set inertia filtering parameters
    params.filterByInertia = True
    params.minInertiaRatio = 0.05

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Initialize array for tuples of x,y coordinates
    xycoord = []

    # Iterate through each frame of video and find the x,y coordinates of
    for i in range(len(video_stack)):

        frame = video_stack[i]

        im = frame.copy()

        # Erosion to filter
        _, im = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY)

        im = 255 - im
        im = 255 - cv2.erode(im, np.ones((3, 3)), iterations=8)

        # Detect blobs
        keypoints = detector.detect(im)

        # Find the center points of each sticker
        for j in range(len(keypoints)):
            xycoord.append((keypoints[j].pt[0], keypoints[j].pt[1]))

    # Find min_x, min_y, max_x, max_y for cropping whole video
    min_x = min(xycoord, key=lambda t: t[0])[0]
    min_y = min(xycoord, key=lambda t: t[1])[1]
    max_x = max(xycoord, key=lambda t: t[0])[0]
    max_y = max(xycoord, key=lambda t: t[1])[1]

    return min_x, min_y, max_x, max_y


def sticker_detection_plot(filename):
    # Load image
    im = cv2.imread(filename, cv2.IMREAD_COLOR)

    # Erode the image (filtering)
    im_orig = im

    _, im = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY)

    im = 255 - im
    im = 255 - cv2.erode(im, np.ones((3, 3)), iterations=8)

    # Set our filtering parameters
    # Initialize parameter setting using cv2.SimpleBlobDetector
    params = cv2.SimpleBlobDetector_Params()

    # Set Area filtering parameters
    params.filterByArea = False
    params.maxArea = 1000

    # Set Circularity filtering parameters
    params.filterByCircularity = False
    params.minCircularity = 0.3

    # Set Convexity filtering parameters
    params.filterByConvexity = True
    params.minConvexity = 0.2

    # Set inertia filtering parameters
    params.filterByInertia = True
    params.minInertiaRatio = 0.05

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(im)

    # Find the center points of each sticker & indicate with small black circles
    xycoord = []

    for idx in range(len(keypoints)):
        xycoord.append((keypoints[idx].pt[0], keypoints[idx].pt[1]))
        cv2.circle(im_orig, (int(keypoints[idx].pt[0]), int(keypoints[idx].pt[1])), 10, (0, 0, 0), 10)

    # Draw blobs on our image as red circles
    blobs = cv2.drawKeypoints(im_orig, keypoints, np.array([]), (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show blobs
    cv2.imshow("Filtering Red Stickers", blobs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return


sticker_detection_plot(file)
