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

    lower1 = np.array([50, 50, 50], dtype="uint8")  # Use 100 for the third, for stationary
    upper1 = np.array([90, 255, 255], dtype="uint8")

    lower2 = np.array([170, 100, 100], dtype="uint8")
    upper2 = np.array([179, 255, 255], dtype="uint8")

    mask1 = cv2.inRange(img_hsv, lower1, upper1)
    #mask2 = cv2.inRange(img_hsv, lower2, upper2)

    output = cv2.bitwise_and(im, im, mask=mask1)

    # captured_frame_lab_red = cv2.inRange(captured_frame_lab, np.array([0, 150, 150]), np.array([10, 255, 255]))

    # Second blur to reduce more noise, easier circle detection
    output = cv2.GaussianBlur(output, (5, 5), 2, 2)

    output = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

    # Use the Hough transform to detect circles in the image
    circles = cv2.HoughCircles(output, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=18, minRadius=10, maxRadius=300)

    # Initialize empty list for radius and center coordinates of each circle
    radii = []
    coords = []

    # If we have extracted a circle, draw an outline
    # We only need to detect one circle here, since there will only be one reference object
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        circles = sorted(circles, key=lambda x: x[0])
        for idx in range(len(circles)):
            cv2.circle(im_orig, center=(circles[idx][0], circles[idx][1]), radius=circles[idx][2], color=(0, 0, 255),
                       thickness=5)
            radii.append(circles[idx][2])
            coords.append((circles[idx][0], circles[idx][1]))

    #cv2.imshow('frame', im_orig)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return radii, coords


def pxl_to_dist(sticker_radius, pixel_radius):
    """
    To automatically calculate the distance scale, we find a ratio (unit: inches/pixels) of the sticker diameter (in inches) to sticker diameter (in pixels)
    """

    ratio = sticker_radius / pixel_radius

    return ratio


def calc_distance_stat(pxl_ratio, coords):
    """
    Given the ratio between distance and pixels, and a set of coordinates, returns distance between all coordinates.
    :param pxl_ratio: output of pxl_to_dist, equivalent to diameter(inches)/diameter(pixels)
    :param coords: list of sticker centers as (x,y) pixel values
    :return: distances list of distances between points, list of coord pairs corresponding to those distances
    """

    # Top left & bottom
    top_left_x = coords[0][0]
    top_left_y = coords[0][1]
    bottom_x = coords[1][0]
    bottom_y = coords[1][1]
    dist1 = pxl_ratio * np.sqrt((top_left_x - bottom_x) ** 2 + (top_left_y - bottom_y) ** 2)  # inches

    # Top right & bottom
    top_right_x = coords[2][0]
    top_right_y = coords[2][1]
    dist2 = pxl_ratio * np.sqrt((top_right_x - bottom_x) ** 2 + (top_right_y - bottom_y) ** 2)  # inches

    # Top left * Top right
    dist3 = pxl_ratio * np.sqrt((top_left_x - top_right_x) ** 2 + (top_left_y - top_right_y) ** 2)  # inches

    return dist1, dist2, dist3


# Distance between circle 1 and 2: 6.75 inches
# Distance between circle

def scale_stat_exp():
    # Experiment 1: three red stickers on paper, 10 inches away
    file1 = '/Users/sang-hyunlee/Desktop/JVP pics/sticker1.jpeg'
    radii1, coords1 = sticker_detection_2(file1)
    pxl_ratio1 = (1 / 2) / radii1[0]
    dist1a, dist1b, dist1c = calc_distance_stat(pxl_ratio1, coords1)

    # Experiment 2: three red stickers on paper, 20 inches away
    file2 = '/Users/sang-hyunlee/Desktop/JVP pics/sticker2.jpeg'
    radii2, coords2 = sticker_detection_2(file2)
    pxl_ratio2 = (1 / 2) / radii2[0]
    dist2a, dist2b, dist2c = calc_distance_stat(pxl_ratio2, coords2)

    # Experiment 3: three red stickers on paper, 30 inches away
    file3 = '/Users/sang-hyunlee/Desktop/JVP pics/sticker3.jpeg'
    radii3, coords3 = sticker_detection_2(file3)
    pxl_ratio3 = (1 / 2) / radii3[0]
    dist3a, dist3b, dist3c = calc_distance_stat(pxl_ratio3, coords3)

    # Experiment 4: three red stickers on paper, 40 inches away
    file4 = '/Users/sang-hyunlee/Desktop/JVP pics/sticker4.jpeg'
    radii4, coords4 = sticker_detection_2(file4)
    pxl_ratio4 = (1 / 2) / radii4[0]
    dist4a, dist4b, dist4c = calc_distance_stat(pxl_ratio4, coords4)

    # Experiment 5: three red stickers on paper, 50 inches away
    file5 = '/Users/sang-hyunlee/Desktop/JVP pics/sticker5.jpeg'
    radii5, coords5 = sticker_detection_2(file5)
    pxl_ratio5 = (1 / 2) / radii5[0]
    dist5a, dist5b, dist5c = calc_distance_stat(pxl_ratio5, coords5)

    # Plotting the results for stationary stickers
    dista = np.array([dist1a, dist2a, dist3a, dist4a, dist5a])
    distb = np.array([dist1b, dist2b, dist3b, dist4b, dist5b])
    distc = np.array([dist1c, dist2c, dist3c, dist4c, dist5c])

    dista_mean = np.mean(dista)
    distb_mean = np.mean(distb)
    distc_mean = np.mean(distc)

    dista_std = np.std(dista)
    distb_std = np.std(distb)
    distc_std = np.std(distc)

    pairs = ['A & B', 'B & C', 'A & C']
    x_pos = np.arange(len(pairs))
    CTEs = [dista_mean, distb_mean, distc_mean]
    error = [dista_std, distb_std, distc_std]

    fig, ax = plt.subplots()
    ax.bar(x_pos + 0.175, CTEs, yerr=error, alpha=0.5, width=0.35, ecolor='black', capsize=10, label='Scaled')
    ax.bar(x_pos - 0.175, [6.875, 6.375, 6.375], width=0.35, label='Actual')
    ax.set_ylabel('Distance (inches)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(pairs)
    ax.set_title('Comparison between real-distance and scaled pixel distance: stationary object (n=5)')
    ax.set_ylim([0, 8])
    ax.legend()

    plt.savefig('scale_plot_stationary.png')
    plt.show()


def scale_human_exp():
    # Experiment 1: three red stickers on paper, 10 inches away
    file1 = '/Users/sang-hyunlee/Desktop/JVP pics/human1.jpg'
    radii1, coords1 = sticker_detection_2(file1)
    pxl_ratio1 = 0.437 / radii1[0]
    dist1a, dist1b, dist1c = calc_distance_stat(pxl_ratio1, coords1)

    # Experiment 2: three red stickers on paper, 20 inches away
    file2 = '/Users/sang-hyunlee/Desktop/JVP pics/human2.jpg'
    radii2, coords2 = sticker_detection_2(file2)
    pxl_ratio2 = 0.437 / radii2[0]
    dist2a, dist2b, dist2c = calc_distance_stat(pxl_ratio2, coords2)

    # Experiment 3: three red stickers on paper, 30 inches away
    file3 = '/Users/sang-hyunlee/Desktop/JVP pics/human3.jpg'
    radii3, coords3 = sticker_detection_2(file3)
    pxl_ratio3 = 0.437 / radii3[0]
    dist3a, dist3b, dist3c = calc_distance_stat(pxl_ratio3, coords3)

    # Experiment 4: three red stickers on paper, 40 inches away
    file4 = '/Users/sang-hyunlee/Desktop/JVP pics/human4.jpg'
    radii4, coords4 = sticker_detection_2(file4)
    pxl_ratio4 = 0.437 / radii4[0]
    dist4a, dist4b, dist4c = calc_distance_stat(pxl_ratio4, coords4)

    # Experiment 5: three red stickers on paper, 50 inches away
    file5 = '/Users/sang-hyunlee/Desktop/JVP pics/human5.jpg'
    radii5, coords5 = sticker_detection_2(file5)
    pxl_ratio5 = 0.437 / radii5[0]
    dist5a, dist5b, dist5c = calc_distance_stat(pxl_ratio5, coords5)

    # Plotting the results for stationary stickers
    dista = np.array([dist1a, dist2a, dist3a, dist4a, dist5a])
    distb = np.array([dist1b, dist2b, dist3b, dist4b, dist5b])
    distc = np.array([dist1c, dist2c, dist3c, dist4c, dist5c])

    dista_mean = np.mean(dista)
    distb_mean = np.mean(distb)
    distc_mean = np.mean(distc)

    dista_std = np.std(dista)
    distb_std = np.std(distb)
    distc_std = np.std(distc)

    pairs = ['A & B', 'B & C', 'A & C']
    x_pos = np.arange(len(pairs))
    CTEs = [dista_mean, distb_mean, distc_mean]
    error = [dista_std, distb_std, distc_std]

    fig, ax = plt.subplots()
    ax.bar(x_pos + 0.175, CTEs, yerr=error, alpha=0.5, width=0.35, ecolor='black', capsize=10, label='Scaled')
    ax.bar(x_pos - 0.175, [5.5, 6.5, 8], width=0.35, label='Actual')
    ax.set_ylabel('Distance (inches)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(pairs)
    ax.set_title('Comparison between real-distance and\nscaled pixel distance: on human (n=5)')
    ax.set_ylim([0, 10])
    ax.legend()

    plt.savefig('scale_plot_human.png')
    plt.show()


#sticker_detection_2('/Users/sang-hyunlee/Desktop/JVP pics/human1.jpg')
#scale_human_exp()
