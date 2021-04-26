import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math


def load_video(vidFile):
    '''
    Reads the video
    :param vidFile: Video file
    :return: video sequence, frame rate, width & height of video frames
    '''
    vid = cv2.VideoCapture(vidFile)
    fr = vid.get(cv2.CAP_PROP_FPS)  # frame rate
    len = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    vidWidth = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    vidHeight = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # save video as stack of images
    video_stack = np.empty((len, vidHeight, vidWidth, 3))

    for x in range(len):
        ret, frame = vid.read()

        video_stack[x] = frame

    vid.release()

    return video_stack, fr, vidWidth, vidHeight


def sticker_detection_3(frame):

    im = (frame.copy()).astype('uint8')

    im_orig = im

    # Red color mask: issue with missing red points
    img_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    lower1 = np.array([50, 50, 50], dtype="uint8")  # Use 100 for the third, for stationary
    upper1 = np.array([90, 255, 255], dtype="uint8")

    mask1 = cv2.inRange(img_hsv, lower1, upper1)

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


def draw_line_on_image(frame):
    radii, coords = sticker_detection_3(frame)
    print(radii, coords)
    pxl_ratio1 = 0.437 / radii[0]
    #im = image.imread(filename)
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

    plt.imshow(frame)
    plt.show()

    x, y, d = linebuilder.xs, linebuilder.ys, linebuilder.inches
    print(f"The first point's coordinates are ({round(x[0], 2)}, {round(y[0], 2)}).")
    print(f"The second point's coordinates are ({round(x[1], 2)}, {round(y[1], 2)}).")
    print(f"The line's distance is {round(d, 2)} inches.")


