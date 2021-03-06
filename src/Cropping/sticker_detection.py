import cv2
import numpy as np
from itertools import combinations


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
    diameters = []

    # Iterate through each frame of video and find the x,y coordinates of
    for i in range(1):

        frame = video_stack[i]

        im = (frame.copy()).astype('uint8')

        # Erosion to filter
        _, im = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY)

        im = 255 - im
        im = 255 - cv2.erode(im, np.ones((3, 3)), iterations=8)

        # Detect blobs
        keypoints = detector.detect(im)

        # Find the center points of each sticker
        for j in range(len(keypoints)):
            xycoord.append((keypoints[j].pt[0], keypoints[j].pt[1]))
            diameters.append(keypoints[j].size)

    # Find min_x, min_y, max_x, max_y for cropping whole video
    min_x = int(min(xycoord, key=lambda t: t[0])[0])
    min_y = int(min(xycoord, key=lambda t: t[1])[1])
    max_x = int(max(xycoord, key=lambda t: t[0])[0])
    max_y = int(max(xycoord, key=lambda t: t[1])[1])

    # Find diameter of neck sticker on first frame (add if detected == 3?)
    diameter = diameters[1]

    return min_x, min_y, max_x, max_y, diameter


def sticker_detection_coords_2(c, video_stack):

    # Initialize empty list for radius and center coordinates of each circle
    radii = []
    coords = []

    # Initialize empty dictionary for {index: [circle coordinates + radius]}
    radii_coords = {}

    # Iterate through each frame of videostack
    for i in range(len(video_stack)):

        # Load each frame
        frame = video_stack[i]
        im = (frame.copy()).astype('uint8')

        # Red color mask: issue with missing red points
        img_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

        if c == 0:  # corresponds to green
            lower1 = np.array([50, 50, 50], dtype="uint8")  # Use 100 for the third, for stationary
            upper1 = np.array([90, 255, 255], dtype="uint8")
        else:  # c == 1, corresponds to blue
            lower1 = np.array([100, 50, 50], dtype="uint8")  # Use 100 for the third, for stationary
            upper1 = np.array([140, 255, 255], dtype="uint8")

        #lower2 = np.array([170, 155, 160], dtype="uint8")
        #upper2 = np.array([179, 255, 255], dtype="uint8")

        mask1 = cv2.inRange(img_hsv, lower1, upper1)
        #mask2 = cv2.inRange(img_hsv, lower2, upper2)

        #output = cv2.bitwise_and(im, im, mask=(mask1 | mask2))
        output = cv2.bitwise_and(im, im, mask=mask1)

        # Second blur to reduce more noise, easier circle detection
        output = cv2.GaussianBlur(output, (5, 5), 2, 2)

        output = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

        # Use the Hough transform to detect circles in the image
        #circles = cv2.HoughCircles(output, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=18, minRadius=5, maxRadius=200) for stationary
        circles = cv2.HoughCircles(output, cv2.HOUGH_GRADIENT, 1, 100, param1=50, param2=18, minRadius=5, maxRadius=300)

        circles = circles[0]

        # Order by pixel y coordinate (2nd one for sternum) 
        if len(circles) == 3:
            circles = np.round(circles).astype("int")
            circles = sorted(circles, key=lambda x: x[1])
            for idx in range(len(circles)):
                radii.append(circles[idx][2])
                coords.append((circles[idx][0], circles[idx][1]))

            radii_coords[i] = circles

        """
        # Order by pixel x coordinate (3rd one for sternum)
        if len(circles) == 3:
            circles = np.round(circles).astype("int")
            circles = sorted(circles, key=lambda x: x[0])
            for idx in range(len(circles)):
                radii.append(circles[idx][2])
                coords.append((circles[idx][0], circles[idx][1]))

            radii_coords[i] = circles
        """

    # Find min_x, min_y, max_x, max_y for cropping whole video
    min_x = int(min(coords, key=lambda t: t[0])[0])
    min_y = int(min(coords, key=lambda t: t[1])[1])
    max_x = int(max(coords, key=lambda t: t[0])[0])
    max_y = int(max(coords, key=lambda t: t[1])[1])

    return min_x, min_y, max_x, max_y, radii_coords


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
    diameters = []

    for idx in range(len(keypoints)):
        xycoord.append((keypoints[idx].pt[0], keypoints[idx].pt[1]))
        cv2.circle(im_orig, (int(keypoints[idx].pt[0]), int(keypoints[idx].pt[1])), 10, (0, 0, 0), 10)
        diameters.append(keypoints[idx].size)

    # Draw blobs on our image as red circles
    blobs = cv2.drawKeypoints(im_orig, keypoints, np.array([]), (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show blobs
    cv2.imshow("Filtering Red Stickers", blobs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return np.array(diameters), np.array(xycoord)


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


if __name__ == "__main__":
    root = '/Users/royphillips/Documents/Rice/elec494/S21/test_videos/'
    file = root + 'red_dots_img.jpeg'
    diameters, xycoord = sticker_detection_plot(file)

    sticker_diam = 1
    pxl_ratio = pxl_to_dist(sticker_diam, diameters[2])


    distances, pair_combos = calc_distance(pxl_ratio,xycoord[:3,:])

    print(distances)
    print(pair_combos)

# [263.0073547363281, 238.82835388183594, 371.4798278808594, 161.4182891845703, 155.4551239013672, 216.84242248535156]
# [(1408.520751953125, 1496.6676025390625), (989.9877319335938, 1175.0413818359375), (923.7012329101562, 774.256103515625), (1518.4801025390625, 669.0260009765625), (192.65757751464844, 422.47564697265625), (1488.2816162109375, 107.33660125732422)]

# Experiment for the stationary sticker distance scale