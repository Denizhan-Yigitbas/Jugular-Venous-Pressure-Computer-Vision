
import numpy as np
import scipy.signal
import wave


def find_peaks(signal):
    # Unique peaks must have at least a separation of this many samples
    peak_sep = 10000

    # Convert to positive signal and compute the minimum peak height
    sig_abs = np.abs(signal)
    threshold = np.max(sig_abs) // 2

    # Calculate the peak indices
    peak_indices, _ = scipy.signal.find_peaks(
        x=sig_abs,
        prominence=threshold,
        distance=peak_sep,
    )

    return peak_indices


def find_peaks_from_file(path):
    raw = wave.open(path)

    # reads all the frames
    # -1 indicates all or max frames
    signal = raw.readframes(-1)
    signal = np.frombuffer(signal, dtype="int16")

    return find_peaks(signal)


def mark_video(vid_array, audio_arr):
    # Size of black square, in pixels
    square_size = 15

    # Get the peaks
    peaks = find_peaks(audio_arr)

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