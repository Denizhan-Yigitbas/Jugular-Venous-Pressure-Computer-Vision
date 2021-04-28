from itertools import combinations

import matplotlib.pyplot as plt
from matplotlib import image
import cv2
import numpy as np
import math
import heapq

from src.EVM_Python.crop_video import sticker_coord_calibration


class LineBuilder:
    def __init__(self, line, ratio):
        self.line = line
        self.ratio = ratio
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.xs.pop()
        self.ys.pop()
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)
        self.distance = 0

    def __call__(self, event):
        # print('click', event)
        if event.inaxes != self.line.axes: return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        if len(self.xs) == 2:
            self.line.set_data(self.xs, self.ys)
            self.line.figure.canvas.draw()
            self.distance = math.sqrt((self.xs[0] - self.xs[1]) ** 2 + (self.ys[0] - self.ys[1]) ** 2)
            self.inches = self.distance * self.ratio
            ax.annotate(f'Line distance is {round(self.inches, 2)} inches', xy=(260, 20), xycoords='figure pixels')
            plt.savefig('testimage.png')
            ax.set_title('Click anywhere on the image to exit')
        if len(self.xs) == 3:
            self.xs.pop()
            self.ys.pop()
            plt.close(fig)


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
    # mask2 = cv2.inRange(img_hsv, lower2, upper2)

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

    # cv2.imshow('frame', im_orig)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return radii, coords


def pxl_to_dist(sticker_radius, pixel_radius):
    """
    To automatically calculate the distance scale, we find a ratio (unit: cm/pixels) of the sticker diameter (in cm) to sticker diameter (in pixels)
    Input sticker_radius is in inches
    """

    ratio = (sticker_radius * 2.45) / pixel_radius

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


def draw_line_on_image(filename):
    radii, coords = sticker_detection_2(filename)
    print(radii, coords)
    pxl_ratio1 = 0.437 / radii[0]
    im = image.imread(filename)
    global fig, ax
    fig, ax = plt.subplots()
    ax.set_title('Click at Top of JVP and on the Left of the Sternum Circle')
    line, = ax.plot([0], [0])  # empty line

    frame1 = plt.gca()
    for xlabel_i in frame1.axes.get_xticklabels():
        xlabel_i.set_visible(False)
        xlabel_i.set_fontsize(0.0)
    for xlabel_i in frame1.axes.get_yticklabels():
        xlabel_i.set_fontsize(0.0)
        xlabel_i.set_visible(False)
    for tick in frame1.axes.get_xticklines():
        tick.set_visible(False)
    for tick in frame1.axes.get_yticklines():
        tick.set_visible(False)

    linebuilder = LineBuilder(line, pxl_ratio1)

    plt.imshow(im)
    plt.show()

    x, y, d = linebuilder.xs, linebuilder.ys, linebuilder.inches
    print(f"The first point's coordinates are ({round(x[0], 2)}, {round(y[0], 2)}).")
    print(f"The second point's coordinates are ({round(x[1], 2)}, {round(y[1], 2)}).")
    print(f"The line's distance is {round(d, 2)} inches.")


### DISTANCE SCALE ON VIDEO PART
def sticker_detection_coords_frame(frame):
    # Initialize empty list for radius and center coordinates of each circle
    radii = []
    coords = []

    # Load each frame
    im = (frame.copy()).astype('uint8')

    # Green color mask
    img_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    lower1 = np.array([50, 50, 50], dtype="uint8")  # Use 100 for the third, for stationary
    upper1 = np.array([90, 255, 255], dtype="uint8")

    mask1 = cv2.inRange(img_hsv, lower1, upper1)

    # output = cv2.bitwise_and(im, im, mask=(mask1 | mask2))
    output = cv2.bitwise_and(im, im, mask=mask1)

    # Second blur to reduce more noise, easier circle detection
    output = cv2.GaussianBlur(output, (5, 5), 2, 2)

    # cv2.imshow('output', output)
    # cv2.waitKey(0)

    output = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

    # Use the Hough transform to detect circles in the image
    circles = cv2.HoughCircles(output, cv2.HOUGH_GRADIENT, 1, 100, param1=40, param2=18, minRadius=5, maxRadius=300)

    # If we have extracted a circle, draw an outline
    # We only need to detect one circle here, since there will only be one reference object
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        circles = sorted(circles, key=lambda x: x[0])
        for idx in range(len(circles)):
            radii.append(circles[idx][2])
            coords.append((circles[idx][0], circles[idx][1]))

    return radii, coords


def draw_scale_on_video(frame, radii, coords):
    # ORDER BY Y INSTEAD OF X????

    sternum = coords[2]

    scale = pxl_to_dist(0.437, radii[0])

    # Main line of scale
    frame = cv2.line(frame, sternum, (sternum[0], 0), (255, 0, 0), 5)

    # Scale ticks
    # Find the number of ticks on the line
    one_tick_length = round(1 / scale)  # 1 cm tick in pixel length, rounded to closest integer
    line_length = sternum[1]
    num_ticks = int(np.floor(line_length / one_tick_length))

    # Draw ticks (1cm apart)
    for idx in range(num_ticks):
        tick_height = sternum[1] - (idx + 1) * one_tick_length
        # Add each tick
        frame = cv2.line(frame, (sternum[0] - 20, tick_height), (sternum[0] + 20, tick_height), (255, 0, 0), 5)

        # Add line that reaches left of pic
        frame = cv2.line(frame, (0, tick_height), (sternum[0] - 70, tick_height), (255, 0, 0), 1)

        # Add label (? cm)
        frame = cv2.putText(frame, str(idx + 1) + ' cm', (sternum[0] - 70, tick_height), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, color=(255, 255, 255), thickness=2)

    cv2.imwrite('/Users/sang-hyunlee/Desktop/JVP pics/cropped_human_ticks.png', frame)


def draw_scale_on_video2(frame, scale, sternum):
    # Main line of scale
    frame = cv2.line(frame, sternum, (sternum[0], 0), (255, 0, 0), 5)

    # Scale ticks
    # Find the number of ticks on the line
    one_tick_length = round(1 / scale)  # 1 cm tick in pixel length, rounded to closest integer
    line_length = sternum[1]
    num_ticks = int(np.floor(line_length / one_tick_length))

    # Draw ticks (1cm apart)
    for idx in range(num_ticks):
        tick_height = sternum[1] - (idx + 1) * one_tick_length
        # Add each tick
        frame = cv2.line(frame, (sternum[0] - 20, tick_height), (sternum[0] + 20, tick_height), (255, 0, 0), 5)

        # Add line that reaches left of pic
        frame = cv2.line(frame, (0, tick_height), (sternum[0] - 70, tick_height), (255, 0, 0), 1)

        # Add label (? cm)
        frame = cv2.putText(frame, str(idx + 1) + ' cm', (sternum[0] - 70, tick_height), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, color=(255, 255, 255), thickness=2)


def average_sternum_position(coords_and_radius, min_x, min_y):

    sternum_x = []
    sternum_y = []

    for stickers in coords_and_radius.keys():
        sternum = sticker_coord_calibration(coords_and_radius[stickers][1], min_x, min_y)
        sternum_x.append(sternum[0])
        sternum_y.append(sternum[1])

    avg_sternum_x = int(np.mean(sternum_x))
    avg_sternum_y = int(np.mean(sternum_y))

    return tuple([avg_sternum_x, avg_sternum_y])


def average_jaw_radius(coords_and_radius):

    jaw_radius = []

    for stickers in coords_and_radius.keys():
        jaw_r = coords_and_radius[stickers][0][2]
        jaw_radius.append(jaw_r)

    avg_jaw_radius = np.mean(jaw_radius)

    return avg_jaw_radius

    # cv2.imwrite('/Users/sang-hyunlee/Desktop/JVP pics/cropped_human_ticks.png', frame)

# frame1 = cv2.imread('/Users/sang-hyunlee/Desktop/JVP pics/cropped_human.png')
# radii, coords = sticker_detection_coords_frame(frame1)

# draw_scale_on_video(frame1, radii, coords)

# sticker_detection_2('/Users/sang-hyunlee/Desktop/JVP pics/human1.jpg')
# scale_human_exp()

# draw_line_on_image('/Users/joshuakowal/Downloads/im_from_p4.png')
