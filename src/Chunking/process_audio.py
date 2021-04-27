
import numpy as np
import scipy.signal
import wave
import moviepy.editor as mp
from scipy.io import wavfile


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

    peaks = np.rint(peaks/(f_audio/2)*f_video).astype(int)
    print("Pulse beeps detected at ", peaks)

    for ind in peaks:

        # Calculate the bounds for the square to be marked out
        min_frame = max(0, ind - 5)
        max_frame = min(vid_array.shape[0], ind + 5)

        min_y = 0
        max_y = min(square_size, vid_array.shape[1])

        min_x = max(0, vid_array.shape[2] - 1 - square_size)
        max_x = vid_array.shape[2] - 1

        # Black out specified region
        print(min_frame,max_frame, min_y,max_y, min_x,max_x)
        vid_array[min_frame:max_frame, min_y:max_y, min_x:max_x, :] = 0

    return vid_array

def mark_video_in_time(vid_array, audio_arr, f_audio, f_video):
    # Size of black square, in pixels
    line_width = 20

    # Get the peaks
    peaks,_ = find_peaks(audio_arr, f_audio)

    peaks = np.rint(peaks/(f_audio/2)*f_video).astype(int)

    total_time = len(audio_arr/f_audio)
    print("Pulse beeps detected at ", peaks)

    for ind in peaks:

        # Calculate the bounds for the square to be marked out
        min_frame = max(0, ind - 5)
        max_frame = min(vid_array.shape[0], ind + 5)

        min_y = 0
        max_y = min(square_size, vid_array.shape[1])

        min_x = max(0, vid_array.shape[2] - 1 - square_size)
        max_x = vid_array.shape[2] - 1

        # Black out specified region
        print(min_frame,max_frame, min_y,max_y, min_x,max_x)
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
    print(fs)
    return fs, audio_array


if __name__ == "__main__":
    import os
    import cv2
    import matplotlib.pyplot as plt
    root = "input_videos/"
    path1 = root + "p4v2.mp4"
    out_path = root + "p4v2.wav"
    fs, audio_arr = video2audio(path1, out_path)

    # video_stack, fr, vidWidth, vidHeight = load_video(path1)

    # print(len(video_stack)/fr, len(audio_arr)/fs)
    # print(len(video_stack), len(audio_arr))
    peaks, filtered = find_peaks(audio_arr, fs)

    # peaks = np.rint(peaks / fs).astype(int)
    print("Pulse beeps detected at ", peaks/fs)

    plt.figure()
    plt.plot(audio_arr)
    plt.title("og")
    plt.figure()
    plt.plot(filtered)
    plt.title("filtered")

    # show_wave_n_spec(out_path)
    # plt.specgram(audio_arr, Fs=fs)

    plt.show()