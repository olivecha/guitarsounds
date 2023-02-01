import struct
import wave
import numpy as np
import scipy.optimize
import scipy.integrate
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
import guitarsounds
import ipywidgets as widgets
from guitarsounds.parameters import sound_parameters

# instantiation of the trim time interpolator
sp = sound_parameters()

freq_dict = {'E2': 82.41,
             'A2': 110.0,
             'D3': 146.83,
             'G3': 196.0,
             'B3': 246.94,
             'E4': 329.63}

trim_dict = {'E2': sp.trim.E2.value,
             'A2': sp.trim.A2.value,
             'D3': sp.trim.D3.value,
             'G3': sp.trim.G3.value,
             'B3': sp.trim.B3.value,
             'E4': sp.trim.E4.value}

freq2trim = interp1d(list(freq_dict.values()),
                     list(trim_dict.values()),
                     fill_value='extrapolate')


def nth_order_polynomial_residual(A, n, x, y):
    """
    Function computing the residual from a nth order
    polynomial, with x and y vectors
    :return: the evaluated residual between a polynomial and the polynomial
    with coefficients A
    """
    y_model = 0
    for i in np.arange(n):
        y_model += A[i] * np.array(x) ** i
    return y_model - y


def nth_order_polynomial_fit(n, x, y):
    """
    Function creating a function of a fitted nth order
    polynomial to a set of x and y data
    :param n: polynomial order
    :param x: x values
    :param y: y values
    :return: the Function such as y = F(x)
    """
    n += 1
    while n > len(x):
        n -= 1
    guess = np.ones(n)
    result = least_squares(nth_order_polynomial_residual,
                                          guess,
                                          args=(n, x, y))
    A = result.x

    def polynomial_function(x_value):
        y_value = 0
        for i, a in enumerate(A):
            y_value += a * np.array(x_value) ** i
        return y_value

    return polynomial_function


def octave_values(fraction, min_freq=10, max_freq=20200):
    """
    Compute octave fraction bin center values from min_freq to max_freq.
    `soundcomp.octave_values(3)` returns the bins corresponding to 1/3 octave
    values
    from 10 Hz to 20200 Hz.
    :param fraction: octave fraction value (1/3, 1, 1/2, ...)
    :param min_freq: minimum frequency in the computed octave fractions
    :param max_freq: maximum frequency considered in the octave fractions
    :return: the value in frequencies corresponding to the octave bins
    """
    # Compute the octave bins
    f = 1000.  # Reference Frequency
    f_up = f
    f_down = f
    multiple = 2 ** (1 / fraction)
    octave_bins = [f]
    while f_up < max_freq or f_down > min_freq:
        f_up = f_up * multiple
        f_down = f_down / multiple
        if f_down > min_freq:
            octave_bins.insert(0, f_down)
        if f_up < max_freq:
            octave_bins.append(f_up)
    return np.array(octave_bins)


def octave_histogram(fraction, **kwargs):
    """
    Compute the octave histogram bins limits corresponding an octave fraction.
    min_freq and max_freq can be provided in the **kwargs.
    :param fraction: octave fraction used to compute the
    octave histogram (1/2, 1/3, ...)
    :**kwargs: key word arguments to pass to the octave_values function
    :return: amplitude of the octave bin histogram
    """
    # get the octave bins
    octave_bins = octave_values(fraction, **kwargs)

    # Compute the histogram bins
    hist_bins = []
    for f_o in octave_bins:
        # Intersecting lower bounds
        hist_bins.append(f_o / 2 ** (1 / (2 * fraction)))
    # Last upper bound
    f_o_end = octave_bins[-1]
    hist_bins.append(f_o_end * 2 ** (1 / (2 * fraction)))
    return np.array(hist_bins)


def trim_sounds(*sounds, length=None):
    """
    Trim sounds to have the same length
    :param sounds: guitarsounds.Sound instances
    :param length: trim length in seconds, if None the sounds a trimmed to the
    length of the shortest one
    :return: the trimmed sounds
    """
    if length is None:
        pack = guitarsounds.SoundPack(*sounds)
        return pack.sounds

    new_sounds = []
    for sound in sounds:
        if length < sound.signal.time()[-1]:
            sound.signal = sound.signal.trim_time(length)
        else:
            raise ValueError("Specify a shorter length")
        sound.bin_divide()
        new_sounds.append(sound)
    return new_sounds


def load_wav(filename):
    """
    load a wave file and return the signal data with the sample rate
    :param filename: string, name of the file to load
    :return: signal_data, sample_rate

    Example :
    y, sr = load_wav('sound.wav')
    """
    audio = wave.open(filename)
    sample_rate = audio.getframerate()
    samples = []
    for _ in range(audio.getnframes()):
        frame = audio.readframes(1)
        samples.append(struct.unpack("h", frame)[0])
    signal = np.array(samples) / 32768
    return signal, sample_rate


def resample(y, sr_orig, sr_target=22050):
    """
    resample a signal using scipy.signal
    :param y: signal data to be resampled
    :param sr_orig: original sample rate
    :param sr_target: target sample rate

    Sample rate = n.o. samples in a second

    Example :
    signal, sr = load_wav('sound.wav')
    # resample to sr=22050
    signal = resample(signal, sr, 22050)
    """
    y_len = int(sr_target * len(y) / sr_orig)
    y_new = scipy.signal.resample(y, num=y_len)
    return y_new


def generate_error_widget(text):
    """
    Creates a ipywidget error message from a string
    :param text: python str of the error message
    :return: widgets.HTLM object corresponding to the error message
    """
    return widgets.HTML('<p style="color:#CC4123;">' + text + '</p>')
