
import numpy as np
import scipy.signal
import wave
import moviepy.editor as mp
from scipy.io import wavfile
import platform

def find_peaks(signal, f_rate):
    # Unique peaks must have at least a separation of this many samples
    peak_sep = 10000

    # sos = scipy.signal.butter(
    #     N=100,
    #     Wn=[2000, 2200],
    #     btype='bandpass',
    #     fs=f_rate,
    #     output='sos',
    # )

    sos = scipy.signal.cheby1(
        N=10,
        rp=5,
        Wn=[11500, 12500],
        btype='bandpass',
        fs=f_rate,
        output='sos',
    )

    filtered = scipy.signal.sosfilt(sos, signal)

    # Convert to positive signal and compute the minimum peak height
    sig_abs = np.abs(filtered)
    threshold = np.max(sig_abs) // 4 * 3

    # Calculate the peak indices
    peak_indices, _ = scipy.signal.find_peaks(
        x=sig_abs,
        prominence=threshold,
        distance=peak_sep,
    )

    return peak_indices, filtered


def find_peaks_from_file(path):
    raw = wave.open(path)

    # reads all the frames
    # -1 indicates all or max frames
    signal = raw.readframes(-1)
    signal = np.frombuffer(signal, dtype="int16")

    f_rate = raw.getframerate()

    peak_indices, _ = find_peaks(signal, f_rate)

    return peak_indices


def mark_video(vid_array, audio_arr, f_audio, f_video):
    # Size of black square, in pixels
    square_size = 100

    # Get the peaks
    peaks,_ = find_peaks(audio_arr, f_audio)
    # print("Pulse beeps detected at seconds", peaks/f_audio)
    peaks = np.rint((peaks-3/f_video*f_audio)/f_audio*f_video).astype(int) # manually shifting audio by 3 frames since 3 are missing in the video from cv2

    for ind in peaks:

        # Calculate the bounds for the square to be marked out
        min_frame = max(0, ind - 5)
        max_frame = min(vid_array.shape[0], ind + 5)

        min_y = 0
        max_y = min(square_size, vid_array.shape[1])

        min_x = max(0, vid_array.shape[2] - 1 - square_size)
        max_x = vid_array.shape[2] - 1

        # Black out specified region
        # print(min_frame,max_frame, min_y,max_y, min_x,max_x)
        vid_array[min_frame:max_frame, min_y:max_y, min_x:max_x, :] = np.array([0, 0, 255])

    return vid_array, peaks


def mark_video_in_time(vid_array, audio_arr, f_audio, f_video):
    # Size of black square, in pixels
    line_width = 20

    # Get the peaks
    peaks,_ = find_peaks(audio_arr, f_audio)

    peaks = np.rint(peaks/(f_audio)*f_video).astype(int)

    total_time = len(audio_arr/f_audio)
    # print("Pulse beeps detected at ", peaks)

    for ind in peaks:

        # Calculate the bounds for the square to be marked out
        min_frame = max(0, ind - 5)
        max_frame = min(vid_array.shape[0], ind + 5)

        min_y = 0
        max_y = min(square_size, vid_array.shape[1])

        min_x = max(0, vid_array.shape[2] - 1 - square_size)
        max_x = vid_array.shape[2] - 1

        # Black out specified region
        vid_array[min_frame:max_frame, min_y:max_y, min_x:max_x, :] = 0

    return vid_array


def video2audio(path, wav_output="temp_audio.wav"):
    """
    converts mp4 video clip to wav output for fft analysis
    """
    my_video = mp.VideoFileClip(path)
    my_video.audio.write_audiofile(wav_output) # output mp4 to wav file
    fs, data = wavfile.read(wav_output)
    audio_array = data.T[1, :] # get 1 channel of audio array
    # print(fs)
    return fs, audio_array
#
# def load_video(vidFile):
#     """
#     Reads the video
#     :param vidFile: Video file
#     :return: video sequence, frame rate, width & height of video frames
#     """
#     vid = cv2.VideoCapture(vidFile)
#     fr = vid.get(cv2.CAP_PROP_FPS)  # frame rate
#     len = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
#     vidWidth = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
#     vidHeight = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
#     # save video as stack of images
#     video_stack = np.empty((len, vidHeight, vidWidth, 3))
#
#     for x in range(len):
#         ret, frame = vid.read()
#
#         video_stack[x] = frame
#
#     vid.release()
#
#     return video_stack, fr, vidWidth, vidHeight
#
# def save_video(video_tensor, fps, filename, var):
#     """
#      Creates a new video for the output
#      :param video_tensor: filtered video sequence
#      :param fps: frame rate of original video
#      :param filename: input video name
#      :param var: variables used in EVM (alpha, cutoff, low, high, linearattenuation, chromattenuation)
#      """
#     path = filename
#     extra = '(alpha-' + str(var[0]) + ', cutoff-' + str(var[1]) + ', low-' + str(var[2]) + ', high-' + str(
#         var[3]) + ', linear-' + str(var[4]) + ', chrom-' + str(var[5]) + ').avi'
#     if platform.system() == 'Linux':
#         fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#     else:
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     [height, width] = video_tensor[0].shape[0:2]
#     writer = cv2.VideoWriter(path[:-4] + extra, fourcc, fps, (width, height), 1)
#
#     for i in range(video_tensor.shape[0]):
#
#         frame = cv2.convertScaleAbs(video_tensor[i])
#
#         frame = cv2.putText(frame, str(i), (300, 50), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=2)
#
#         writer.write(frame)
#
#     writer.release()
#     return extra
# #
#
# import sys
# from pylab import *
# import wave
#
#
# def show_wave_n_spec(speech):
#     spf = wave.open(speech, 'r')
#     sound_info = spf.readframes(-1)
#     sound_info = fromstring(sound_info, 'Int16')
#
#     f = spf.getframerate()
#
#     subplot(211)
#     plot(sound_info)
#     # title('Wave from and spectrogram of %s' % sys.argv[1])
#
#     subplot(212)
#     spectrogram = specgram(sound_info, Fs=f, scale_by_freq=True, sides='default')
#
#     show()
#     spf.close()
#     return sound_info


if __name__ == "__main__":
    import os
    import cv2
    import matplotlib.pyplot as plt
    root = "input_videos/"
    path1 = root + "p4v2.mp4"
    out_path = root + "p4v2.wav"
    filename = root + "temp_video"
    fs, audio_arr = video2audio(path1, out_path)

    video_stack, fr, vidWidth, vidHeight = load_video(path1)

    print("Length in seconds", len(video_stack)/fr, len(audio_arr)/fs)
    print("Frames, samples", len(video_stack), len(audio_arr))
    # peaks, filtered = find_peaks(audio_arr, fs)
    # print("Pulse beeps detected at ", peaks)
    # peaks = np.rint(peaks / (fs / 2) * fr).astype(int)

    final = mark_video(video_stack, audio_arr, fs, fr)

    var = range(6)
    extra = save_video(final, fr, filename, var)
    # peaks = np.rint(peaks / fs).astype(int)
    # print("Pulse beeps detected at ", peaks)

    # plt.figure()
    # plt.plot(audio_arr)
    # plt.title("og")
    # plt.figure()
    # plt.plot(filtered)
    # plt.title("filtered")

    # show_wave_n_spec(out_path)
    # plt.specgram(audio_arr, Fs=fs)

    plt.show()