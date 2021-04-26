import os
import cv2
import numpy as np
from scipy.signal import butter, lfilter
import platform
from scipy import fftpack
import time

from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image

from EVM_Python.crop_video import crop_video
from Cropping.sticker_detection import sticker_detection_coords, pxl_to_dist, sticker_detection_coords_2
from jvp_height import draw_line_on_image

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
target = os.path.join(APP_ROOT, 'UPLOAD_FOLDER/')
if not os.path.isdir(target):
    os.mkdir(target)

UPLOAD_FOLDER = target
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CROPPING_MARGIN = 50


# check if selected file is verified
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            # TODO: What is this doing?
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            # TODO: Think about this?
            # flash('No selected file')
            # return redirect(request.url)
            return render_template('upload.html', no_file_selected=True)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(
                url_for(
                    'uploaded_file',
                    filename=filename,
                    alpha=request.form['alpha'],
                    cutoff=request.form['cutoff'],
                    low=request.form['low'],
                    high=request.form['high'],
                    linearAttenuation=request.form['linearAttenuation'],
                    chromAttenuation=request.form['chromAttenuation'],
                )
            )
    return render_template('upload.html', no_file_selected=False)


@app.route('/get-height/', methods=['GET', 'POST'])
def upload_jvp_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            # TODO: What is this doing?
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            # TODO: Think about this?
            # flash('No selected file')
            # return redirect(request.url)
            return render_template('measure_jvp.html', no_file_selected=True)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(
                url_for(
                    'measure_jvp',
                    filename=filename,
                    frame_num=request.form['frame_num'],
                )
            )
    return render_template('measure_jvp.html', no_file_selected=False)


@app.route('/get-height/uploads/<filename>')
def measure_jvp(filename):

    video_stack, _, _, _ = load_video(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    frame_num = int(request.args.get('frame_num'))
    frame = video_stack[frame_num]

    draw_line_on_image(frame)


    return render_template('measure_jvp.html', no_file_selected=True)


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


def rgb2yiq(video):
    '''
    Converts the video color from RGB to YIQ (NTSC)
    :param video: RGB video sequence
    :return: YIQ-color video sequence
    '''
    yiq_from_rgb = np.array([[0.299, 0.587, 0.114],
                             [0.596, -0.274, -0.322],
                             [0.211, -0.523, 0.312]])
    t = np.dot(video, yiq_from_rgb.T)
    return t


def calculate_pyramid_levels(vidWidth, vidHeight):
    '''
    Calculates the maximal pyramid levels for the Laplacian pyramid
    :param vidWidth: video frames' width
    :param vidHeight: video frames' height
    '''
    if vidWidth < vidHeight:
        levels = int(np.log2(vidWidth))
    else:
        levels = int(np.log2(vidHeight))

    return levels


def create_gaussian_pyramid(image, levels):
    '''
    Creates a Gaussian pyramid for each image.
    :param image: An image, i.e video frame
    :param levels: The Gaussian pyramid level
    :return: Returns a pyramid of nr. levels images
    '''
    gauss = image.copy()
    gauss_pyr = [gauss]

    for level in range(1, levels):
        gauss = cv2.pyrDown(gauss)
        gauss_pyr.append(gauss)

    return gauss_pyr


def gaussian_video(video_tensor, levels):
    '''
    For a given video sequence the function creates a video with
    the highest (specified by levels) Gaussian pyramid level
    :param video_tensor: Video sequence
    :param levels: Specifies the Gaussian pyramid levels
    :return: a video sequence where each frame is the downsampled of the original frame
    '''
    for i in range(0, video_tensor.shape[0]):
        frame = video_tensor[i]
        pyr = create_gaussian_pyramid(frame, levels)
        gaussian_frame = pyr[-1]  # use only highest gaussian level
        if i == 0:  # initialize one time
            vid_data = np.zeros((video_tensor.shape[0], gaussian_frame.shape[0], gaussian_frame.shape[1], 3))

        vid_data[i] = gaussian_frame
    return vid_data


def create_laplacian_pyramid(image, levels):
    '''
    Builds a Laplace pyramid for an image, i.e. video frame
    :param image: Image,  i.e. single video frame
    :param levels: Specifies the Laplace pyramid levels
    :return: Returns a pyramid of nr. levels images
    '''
    gauss_pyramid = create_gaussian_pyramid(image, levels)
    laplace_pyramid = []
    for i in range(levels - 1):
        size = (gauss_pyramid[i].shape[1], gauss_pyramid[i].shape[0])  # reshape
        laplace_pyramid.append(gauss_pyramid[i] - cv2.pyrUp(gauss_pyramid[i + 1], dstsize=size))

    laplace_pyramid.append(gauss_pyramid[-1])  # add last gauss pyramid level
    return laplace_pyramid


def laplacian_video_pyramid(video_stack, levels):
    '''
    Creates a Laplacian pyramid for the whole video sequence
    :param video_stack: Video sequence
    :param levels: Specifies the Laplace pyramid levels
    :return: A two-dimensional array where the first index is used for the pyramid levels
    and the second for each video frame
    '''
    # "2 dimensional" array - first index for pyramid level, second for frames
    laplace_video_pyramid = [[0 for x in range(video_stack.shape[0])] for x in range(levels)]

    for i in range(video_stack.shape[0]):
        frame = video_stack[i]
        pyr = create_laplacian_pyramid(frame, levels)

        for n in range(levels):
            laplace_video_pyramid[n][i] = pyr[n]

    return laplace_video_pyramid


def butter_bandpass(lowcut, highcut, fs, order=1):
    '''
    Calculates the Butterworth bandpass filter
    :param lowcut: low frequency cutoff
    :param highcut: high frequency cutoff
    :param fs: video frame rate
    :param order: filter order - per default = 1
    :return: Numerator (b) and denominator (a) polynomials of the IIR filter.
    '''

    low = lowcut / fs
    high = highcut / fs
    b, a = butter(order, [low, high], btype='band')
    return b, a


def apply_butter(laplace_video_list, levels, alpha, cutoff, low, high, fps, width, height, linearAttenuation):
    '''
    Applies the Butterworth filter on video sequence, magnifies the filtered video sequence
    and cuts off spatial frequencies
    :param laplace_video_list: Laplace video pyramid
    :param levels: Pyramid levels
    :param alpha: Magnification factor
    :param cutoff: Spatial frequencies cutoff factor
    :param low: Temporal low frequency cutoff
    :param high: Temporal high frequency cutoff
    :param fps: Video frame rate
    :param width: Video frame width
    :param height: Video frame height
    :param linearAttenuation: Boolean if linear attenuation should be applied
    :return:
    '''
    filtered_video_list = []
    b, a = butter_bandpass(low, high, fps, order=1)

    # spacial wavelength lambda
    lambda1 = (width ** 2 + height ** 2) ** 0.5

    delta = cutoff / 8 / (1 + alpha)

    for i in range(levels):  # pyramid levels

        current_alpha = lambda1 / 8 / delta - 1  # given in paper
        current_alpha /= 2

        # apply the butterworth filter onto temporal image sequence
        filtered = lfilter(b, a, laplace_video_list[i], axis=0)

        if i == levels - 1 or i == 0:  # ignore lowest and highest level
            filtered *= 0

        # spacial frequencies attenuation
        if current_alpha > alpha:
            filtered *= alpha
        else:
            if linearAttenuation:
                filtered *= current_alpha
            else:
                filtered *= 0

        filtered_video_list.append(filtered)

        lambda1 /= 2

    return filtered_video_list


def ideal_bandpass(laplace_video_list, alpha, low, high, fps):
    print("Applying bandpass between " + str(low) + " and " + str(high) + " Hz")

    fft = fftpack.rfft(laplace_video_list, axis=0)

    frequencies = fftpack.rfftfreq(fft.shape[0], d=1.0 / fps)

    mask = np.logical_and(frequencies > low, frequencies < high)

    fft[~mask] = 0

    filtered = fftpack.irfft(fft, axis=0)

    filtered *= alpha

    return filtered


def reconstruct(filtered_video, levels):
    '''
    Reconstructs a video sequence from the filtered Laplace video pyramid
    :param filtered_video: 2 dimensional video sequence - 1st. index pyramid levels, 2nd. - video frames
    :param levels: pyramid levels
    :return: video sequence
    '''
    final = np.empty(filtered_video[0].shape)
    for i in range(filtered_video[0].shape[0]):  # iterate through frames

        up = filtered_video[-1][i]  # highest level
        for k in range(levels - 1, 0, -1):  # going down to lowest level
            size = (filtered_video[k - 1][i].shape[1], filtered_video[k - 1][i].shape[0])  # reshape
            up = cv2.pyrUp(up, dstsize=size) + filtered_video[k - 1][i]

        final[i] = up

    return final


def yiq2rgb(video):
    '''
    Converts the video color from YIQ (NTSC) to RGB
    :param video: YIQ-color video sequence
    :return: RGB video sequence
    '''
    rgb_from_yiq = np.array([[1, 0.956, 0.621],
                             [1, -0.272, -0.647],
                             [1, -1.106, 1.703]])
    t = np.dot(video, rgb_from_yiq.T)
    return t


def save_video(video_tensor, fps, filename, var):
    """
     Creates a new video for the output
     :param video_tensor: filtered video sequence
     :param fps: frame rate of original video
     :param filename: input video name
     :param var: variables used in EVM (alpha, cutoff, low, high, linearattenuation, chromattenuation)
     """
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    extra = '(alpha-' + str(var[0]) + ', cutoff-' + str(var[1]) + ', low-' + str(var[2]) + ', high-' + str(
        var[3]) + ', linear-' + str(var[4]) + ', chrom-' + str(var[5]) + ').avi'
    if platform.system() == 'Linux':
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    [height, width] = video_tensor[0].shape[0:2]
    writer = cv2.VideoWriter(path + extra, fourcc, fps, (width, height), 1)
    for i in range(video_tensor.shape[0]):

        frame = cv2.convertScaleAbs(video_tensor[i])

        frame = cv2.putText(frame, str(i), (300, 50), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=2)

        writer.write(frame)

    writer.release()
    return extra

"""
def save_video(video_tensor, fps, filename):
    '''
    Creates a new video for the output
    :param video_tensor: filtered video sequence
    :param fps: frame rate of original video
    :param name: output video name
    '''
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if platform.system()=='Linux':
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'PIM1')
    [height, width] = video_tensor[0].shape[0:2]
    writer = cv2.VideoWriter(path+"Out.avi", fourcc, fps, (width, height), 1)
    for i in range(video_tensor.shape[0]):
        writer.write(cv2.convertScaleAbs(video_tensor[i]))
    writer.release()
"""


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    alpha = float(request.args.get('alpha'))
    cutoff = float(request.args.get('cutoff'))
    low = float(request.args.get('low'))
    high = float(request.args.get('high'))
    linearAttenuation = float(request.args.get('linearAttenuation'))
    chromAttenuation = float(request.args.get('chromAttenuation'))

    print(f"Alpha = {alpha}\nCutOff = {cutoff}\nLow = {low}\n"
          f"High = {high}\nLinearAttenuation = {linearAttenuation}\n"
          f"ChromaticAttenuation = {chromAttenuation}\n")

    var = [alpha, cutoff, low, high, linearAttenuation, chromAttenuation]

    print("File Submission Clicked")
    print("Loading Video...")
    start = time.time()
    t, fps, width, height = load_video(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    end = time.time()
    print("Video Loaded in " + str(end - start) + " seconds\n")

    print("Cropping Video...")
    start = time.time()
    # Find the coordinates based on stickers to crop the videos
    min_x, min_y, max_x, max_y, radii = sticker_detection_coords_2(video_stack=t)

    min_x = max(min_x - CROPPING_MARGIN, 0)
    min_y = max(min_y - CROPPING_MARGIN, 0)
    max_x = min(max_x + CROPPING_MARGIN, width)
    max_y = min(max_y + CROPPING_MARGIN, height)

    # TODO: Video preprocessing
    # Insert calls to the circle finding code here to identify cropping targets
    # Fallback (in the event no/partial circles identified) should be base video bounds
    t, width, height = crop_video(
        video_stack=t,
        min_x=min_x,
        min_y=min_y,
        max_x=max_x,
        max_y=max_y,
    )
    end = time.time()
    print("Video cropped in " + str(end - start) + " seconds\n")

    # Find the scale (ratio) between pixel distance & real distance using sticker (unit: inches/pixels)
    #scale = pxl_to_dist(sticker_diameter=1, pixel_diameter=diameter)

    print("Calculating Laplacian Pyramid")
    start = time.time()
    t = rgb2yiq(t)
    levels = calculate_pyramid_levels(width, height)
    end = time.time()
    print("Laplacian Pyramid Calculated in " + str(end - start) + " seconds\n")

    print("Creating Laplacian Video")
    start = time.time()
    lap_video_list = laplacian_video_pyramid(t, levels)
    end = time.time()
    print("Laplacian Video created in " + str(end - start) + " seconds\n")

    print("Applying Butterworth Filter")
    start = time.time()
    filtered_video_list = apply_butter(lap_video_list, levels, alpha, cutoff, low, high, fps, width, height,
                                       linearAttenuation)
    end = time.time()
    print("Butterworth Filter Applied in " + str(end - start) + " seconds\n")

    print("Reconstructing Video")
    start = time.time()
    final = reconstruct(filtered_video_list, levels)
    end = time.time()
    print("Video Reconstructed in " + str(end - start) + " seconds\n")

    # chromatic attenuation
    final[:][:][:][1] *= chromAttenuation
    final[:][:][:][2] *= chromAttenuation

    # Add to original
    final += t

    # from yiq to rgb
    final = yiq2rgb(final)

    # Cutoff wrong values
    final[final < 0] = 0
    final[final > 255] = 255

    print("Saving Video")
    extra = save_video(final, fps, filename, var)
    print("Video Saved!!!")

    # Convert uploaded image to Black and White - REMOVE
    # image_file = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename)) # open colour image
    # image_file = image_file.convert('1') # convert image to black and white
    # image_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    return send_from_directory(
        app.config['UPLOAD_FOLDER'],
        filename + extra,
        as_attachment=True
    )


if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0")
