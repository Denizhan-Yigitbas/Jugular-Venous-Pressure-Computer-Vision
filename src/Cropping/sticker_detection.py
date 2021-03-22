import cv2
import numpy as np

filename = '/Users/sang-hyunlee/Desktop/RotatedCirclescopy.jpeg'

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
blobs = cv2.drawKeypoints(im_orig, keypoints, np.array([]), (255, 0, 0),
                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

"""
number_of_blobs = len(keypoints)
text = "Number of Circular Blobs: " + str(len(keypoints))
cv2.putText(blobs, text, (20, 550),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
"""
# Show blobs
cv2.imshow("Filtering Red Stickers", blobs)
cv2.waitKey(0)
cv2.destroyAllWindows()