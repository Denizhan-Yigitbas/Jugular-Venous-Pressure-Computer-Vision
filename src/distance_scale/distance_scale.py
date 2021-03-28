from itertools import combinations

import matplotlib.pyplot as plt
import cv2
import numpy as np
import heapq


def sticker_detection_plot(filename):
    # Load image
    im = cv2.imread(filename, cv2.IMREAD_COLOR)

    # Erode the image (filtering)
    im_orig = im
    # _, im = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY)

    # im = 255 - im
    # im = 255 - cv2.erode(im, np.ones((3, 3)), iterations=8)

    # Red color mask: issue with missing red points
    img_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 170, 100], dtype="uint8")
    upper1 = np.array([10, 255, 255], dtype="uint8")

    lower2 = np.array([170, 170, 100], dtype="uint8")
    upper2 = np.array([179, 255, 255], dtype="uint8")

    mask1 = cv2.inRange(img_hsv, lower1, upper1)
    mask2 = cv2.inRange(img_hsv, lower2, upper2)

    mask3 = mask1 | mask2

    output = cv2.bitwise_and(im, im, mask=mask3)

    # _, output = cv2.threshold(output, 128, 255, cv2.THRESH_BINARY)

    # output = 255 - output
    # output = 255 - cv2.erode(output, np.ones((3, 3)), iterations=1)

    cv2.imshow("images", np.hstack([im, output]))
    cv2.waitKey(0)

    # Set our filtering parameters
    # Initialize parameter setting using cv2.SimpleBlobDetector
    params = cv2.SimpleBlobDetector_Params()

    # Set Area filtering parameters
    params.filterByArea = False
    params.minArea = 100

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
    keypoints = detector.detect(output)

    # Find the center points of each sticker & indicate with small black circles
    xycoord = []
    diameters = []

    for idx in range(len(keypoints)):
        xycoord.append((keypoints[idx].pt[0], keypoints[idx].pt[1]))
        cv2.circle(im_orig, (int(keypoints[idx].pt[0]), int(keypoints[idx].pt[1])), 10, (0, 0, 0), 10)
        diameters.append(keypoints[idx].size)

    # top_3 = heapq.nlargest(3, zip(diameters, xycoord))

    # for idx in range(len(top_3)):
    #    cv2.circle(im_orig, (int(top_3[idx][1][0]), int(top_3[idx][1][1])), 10, (0, 0, 0), 10)

    # Draw blobs on our image as red circles
    blobs = cv2.drawKeypoints(im_orig, keypoints, np.array([]), (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show blobs
    cv2.imshow("Filtering Red Stickers", blobs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return np.array(diameters), np.array(xycoord)


def sticker_detection_2(filename):
    # Load image
    im = cv2.imread(filename, cv2.IMREAD_COLOR)
    im_orig = im

    # Red color mask: issue with missing red points
    img_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 155, 155], dtype="uint8")         # Use 100 for the third, for stationary
    upper1 = np.array([10, 255, 255], dtype="uint8")

    lower2 = np.array([170, 155, 155], dtype="uint8")
    upper2 = np.array([179, 255, 255], dtype="uint8")

    mask1 = cv2.inRange(img_hsv, lower1, upper1)
    mask2 = cv2.inRange(img_hsv, lower2, upper2)

    output = cv2.bitwise_and(im, im, mask=(mask1 | mask2))

    # captured_frame_lab_red = cv2.inRange(captured_frame_lab, np.array([0, 150, 150]), np.array([10, 255, 255]))

    # Second blur to reduce more noise, easier circle detection
    output = cv2.GaussianBlur(output, (5, 5), 2, 2)

    output = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

    # Use the Hough transform to detect circles in the image
    circles = cv2.HoughCircles(output, cv2.HOUGH_GRADIENT, 1, 30,
                               param1=100, param2=18, minRadius=5, maxRadius=60)

    # If we have extracted a circle, draw an outline
    # We only need to detect one circle here, since there will only be one reference object
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for idx in range(circles.shape[0]):
            cv2.circle(im_orig, center=(circles[idx, 0], circles[idx, 1]), radius=circles[idx, 2], color=(0, 255, 0),
                       thickness=5)

    cv2.imshow('frame', im_orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def pxl_to_dist(sticker_diameter, pixel_diameter):
    """
    To automatically calculate the distance scale, we find a ratio (unit: inches/pixels) of the sticker diameter (in inches) to sticker diameter (in pixels)
    """

    ratio = sticker_diameter / pixel_diameter

    return ratio


def calc_distance(pxl_ratio, coords):
    """
    Given the ratio between distance and pixels, and a set of coordinates, returns distance between all coordinates.
    :param pxl_ratio: output of pxl_to_dist, equivalent to diameter(inches)/diameter(pixels)
    :param coords: list of sticker centers as (x,y) pixel values
    :return: distances list of distances between points, list of coord pairs corresponding to those distances
    """
    distances = []

    coord_combo = combinations(coords, 2)
    coord_pairs = []
    for coords in coord_combo:
        [[x1, y1], [x2, y2]] = coords
        distance = pxl_ratio * np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        distances.append(distance)
        coord_pairs.append(coords)

    return distances, coord_pairs


# Distance between circle 1 and 2: 6.75 inches

# Experiment 1: three red stickers on paper, 10 inches away
file1 = '/Users/sang-hyunlee/Desktop/sticker1.jpeg'

# Experiment 2: three red stickers on paper, 20 inches away
file2 = '/Users/sang-hyunlee/Desktop/human1.png'

# Experiment 3: three red stickers on paper, 30 inches away

# Experiment 4: three red stickers on paper, 40 inches away

# Experiment 5: three red stickers on paper, 50 inches away


# files = [file1, file2, file3, file4, file5]

sticker_detection_2(file2)
