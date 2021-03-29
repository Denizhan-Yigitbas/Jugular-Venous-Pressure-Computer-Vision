
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