from scipy import fftpack
import scipy
import numpy as np


def temporal_bandpass_filter(data, fps, fl, fh):

    axis = 0

    print("Applying bandpass between " + str(fl) + " and " + str(fh) + " Hz")
    fft = scipy.fftpack.rfft(data, axis=axis)

    frequencies = scipy.fftpack.fftfreq(len(data), d=1.0 / fps)

    mask = np.logical_and(frequencies > fl, frequencies < fh)  # logical array if values between low and high frequencies

    fft[~mask] = 0  # cutoff values outside the bandpass

    filtered = fftpack.irfft(fft, axis=0)  # inverse fourier transformation

    #filtered *= alpha  # magnification

    # chromatic attenuation
    #filtered[:][:][:][1] *= chromAttenuation
    #filtered[:][:][:][2] *= chromAttenuation

    #bound_low = (np.abs(frequencies - fl)).argmin()
    #bound_high = (np.abs(frequencies - fh)).argmin()

    #fft[:bound_low] = 0
    #fft[bound_high:-bound_high] = 0
    #fft[-bound_low:] = 0

    #result = []
    #np.ndarray(shape=, dtype='float')
    #result[:] = scipy.fftpack.ifft(fft, axis=0)
    #result *= alpha



    return filtered



