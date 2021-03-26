import numpy as np
import cv2
# import matplotlib.pyplot as plt
import moviepy.editor as mp
import copy
from scipy.io import wavfile
from multiprocessing import Process, Semaphore


def video2audio(path, wav_output):
    """
    converts mp4 video clip to wav output for fft analysis
    """
    my_video = mp.VideoFileClip(path)
    my_video.audio.write_audiofile(wav_output) # output mp4 to wav file
    return my_video

def find_fft(path2file, timestep=1/60, chan=0):
    """
    returns the frequencies and fft value of the wav file at the given path
    """
    fs, data = wavfile.read(path2file)  # load the data
    chan1 = data.T[chan, :]
    # chan2 = data.T[1, :]
    fft_arry = np.fft.fft(chan1)  # calculate fourier transform (complex numbers list)
    lenc = len(fft_arry)  # if memory a problem I should only need half of the fft list (real signal symmetry)
    freq = np.fft.fftfreq(lenc, d=timestep)
    freq_hz = freq * 1 / timestep

    return fft_arry, freq_hz, data, fs

def remove_beeps(freq_hz, fft_arry, og_data, beep_tone=1000, fwindow_len =250):
    """
    bandpass filters signal around the beep and compares the difference from the filtered signal
    returns indices of chunks and chunked audio file
    """

    # find positive and negative frequency beeps (symmetric since real signal)
    beep_idx1 = np.argwhere(abs(freq_hz - beep_tone) < fwindow_len)
    beep_idx2 = np.argwhere(abs(freq_hz + beep_tone) < fwindow_len)
    filtered_fft = copy.deepcopy(fft_arry)
    filtered_fft[beep_idx1] = 0
    filtered_fft[beep_idx2] = 0
    without_beep = np.around(np.fft.ifft(filtered_fft).real, 2)  # round the ifft to match original

    # find the difference
    difference = og_data - without_beep


    # find beep indices
    beep_timedx = np.argwhere(difference > 100) # location of beep times
    starting_beep_timedx = np.argwhere(np.diff(beep_timedx, 1, 0) > 16)[:, 0] # the beginning of each beep zone
    starting_timedx = beep_timedx[starting_beep_timedx] # moving indices back to original values
    mid_idx = (starting_timedx[:-1] + starting_timedx[1:])//2 # find midpoint between pulses

    # chunk audio file
    chunked = np.split(og_data, mid_idx[:, 0])

    return mid_idx, chunked

def write_videofile(my_clip, clip_start, clip_end, output_path, n, pool_sema=1):
    """
    split video file and write
    """
    clip = my_clip.subclip( clip_start, clip_end)
    print( clip_start, clip_end)
    clip.write_videofile(output_path, fps=60, bitrate="4000k",
                             threads=1, preset='ultrafast', codec='h264')
    # except:
    #     print("error",  clip_start, clip_end)
    # finally:
    # pool_sema.release()


def write_chunks(my_clip, chunk_idx, fs, testroot):
    """
    call write_video to chunk my_clip on chunk_idx
    """
    # pool_sema = Semaphore(6)
    for n, chunk in enumerate(chunk_idx):
        if chunk == chunk_idx[-1]:
            break

        clip_start = chunk / fs
        clip_end = chunk_idx[n+1] / fs
        # pool_sema.acquire()
        # p = Process(target=write_videofile, args=(my_clip, clip_start, clip_end, n, pool_sema)).start()
        write_videofile(my_clip, clip_start, clip_end, testroot+'chunk'+str(n)+'.mov', n)

if __name__ == "__main__":
    # run code

    # path to file
    root = "/Users/royphillips/Documents/Rice/elec494/S21/test_videos/"
    path1 = root + "/red_dots_sound.mp4"
    path2 = root + "/Stationaries/black_background.mov"
    testroot = root + 'wav_testing/'
    wav_output = testroot + "output_audio.wav"
    wav_test_output = testroot + "test_output.wav"

    my_video = video2audio(path1, wav_output)

    chan = 0
    fft_arry, freq_hz, data, fs = find_fft(wav_output, chan=chan)
    print(fs)
    mid_idx, chunked = remove_beeps(freq_hz, fft_arry, data.T[chan, :])

    # split video clip
    write_chunks(my_video, mid_idx, fs,testroot)
