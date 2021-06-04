import librosa
import librosa.display
from soundfile import write
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
import os
from noisereduce import reduce_noise
import scipy
from scipy import signal as sig

"""
Global functions
"""


def double_plot(plot1, plot2, **kwargs):
    """
    Plot two plots side by side **kwargs pass in the plotting arguments, such as kind.
    Plots are the same
    Useful to compare data from two different sounds
    Example :
    soundcomp.double_plot(Sound1.signal.plot, Sound2.signal.plot, kind='envelop')
    """
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plot1(**kwargs)
    plt.subplot(1, 2, 2)
    plot2(**kwargs)

def octave_values(fraction, min_freq=10, max_freq=20200):
    """
    Compute octave fraction bin center values from min_freq to max_freq.
    `soundcomp.octave_values(3)` returns the bins corresponding to 1/3 octave values
    from 10 Hz to 20200 Hz.
    """
    # Compute the octave bins
    f = 1000.  # fréquence de référence
    f_up = f
    f_down = f
    multiplicator = 2 ** (1 / fraction)
    octave_bins = [f]
    while f_up < max_freq or f_down > min_freq:
        f_up = f_up * multiplicator
        f_down = f_down / multiplicator
        if f_down > min_freq:
            octave_bins.insert(0, f_down)
        if f_up < max_freq:
            octave_bins.append(f_up)
    return np.array(octave_bins)

def octave_histogram(fraction, **kwargs):
    """
    Compute the octave histogram bins limits corresponding an octave fraction.
    min_freq and max_freq can be provided in the **kwargs.
    """
    # get the octave bins
    octave_bins = octave_values(fraction, **kwargs)

    # Compute the histogram bins
    hist_bins = []
    for f_o in octave_bins:
        # Bornes inférieures qui s'intersectent
        hist_bins.append(f_o / 2 ** (1 / (2 * fraction)))
    # La dernière borne supérieure
    hist_bins.append(f_o * 2 ** (1 / (2 * fraction)))
    return np.array(hist_bins)

def power_split(signal, x, xmax, nbins):
    """ Return the index of the split bins of a signal FT with equal integrals"""
    imax = np.where(x > xmax)[0][0]
    A = scipy.integrate.trapezoid(signal[:imax])
    I = A / nbins
    indexes = [0] * nbins
    for i in range(nbins - 1):
        i1 = indexes[i]
        i2 = i1 + 1
        while scipy.integrate.trapezoid(signal[i1:i2]) < I:
            i2 += 1
        indexes[i + 1] = i2
    return indexes

def time_compare(*sons, fbin='all'):
    """
    A function to compare signals decomposed frequency wise in the time domain on a logarithm scale.
    The functions take n sounds and plots their frequency bins according to the frequency bin argument fbin.

    Supported arguments are : 'all', 'bass', 'mid', 'highmid', 'uppermid', 'presence', 'brillance'
    """
    if fbin == 'all':
        # plot chaque graph pour toutes les bins
        for key in sons[0].bins.keys():
            _ = plt.figure(figsize=(10, 8))
            # plot chaque son dans l'appel de la fonction
            for i, son in enumerate(sons):
                lab = ' ' + key + ' : ' + str(int(son.bins[key].range[0])) + ' - ' + str(
                    int(son.bins[key].range[1])) + ' Hz'
                son.bins[key].normalise().plot('log envelop', label=(str(i + 1) + '. ' + son.name + lab))
                plt.xscale('log')
                plt.legend()
    elif fbin in sons[0].bins.keys():
        plt.figure(figsize=(10, 8))
        # plot chaque son dans l'appel de la fonction
        for i, son in enumerate(sons):
            lab = ' ' + fbin + ' : ' + str(int(son.bins[fbin].range[0])) + ' - ' + str(
                int(son.bins[fbin].range[1])) + ' Hz'
            son.bins[fbin].normalise().plot('log envelop', label=(str(i + 1) + '. ' + son.name + lab))
            plt.xscale('log')
            plt.legend()
    else:
        print('fbin invalid')

def fft_mirror(son1, son2, max_freq=4000):
    """ Plot the fourier transforms of two signals on the y and -y axes to compare them"""
    index = np.where(son1.signal.fft_freqs > max_freq)[0][0]
    plt.figure(figsize=(10, 8))
    plt.yscale('symlog')
    plt.grid('on')
    plt.plot(son1.signal.fft_freqs[:index], son1.signal.fft[:index], label='1 : ' + son1.name)
    plt.plot(son2.signal.fft_freqs[:index], -son2.signal.fft[:index], label='2 : ' + son2.name)
    plt.xlabel('fréquence (Hz)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

def fft_diff(son1, son2, fraction=3):
    """
    Compare the Fourier Transform of two sounds by computing the diffenrence of the octave bins heights.
    The two FTs are superimposed on the first plot to show differences
    The difference between the two FTs is plotted on the second plot
    The `fraction` value corresponds to the octave bin fraction.
    A higher number will show a more precise comparison but conclusions may be harder to draw.
    """
    # Compute plotting bins
    x_values = octave_values(fraction)
    hist_bins = octave_histogram(fraction)
    bar_widths = np.array([hist_bins[i + 1] - hist_bins[i] for i in range(0, len(hist_bins) - 1)])

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Histogramme de la FT des deux sons')
    plot1 = plt.hist(son1.signal.fft_bins(), octave_histogram(fraction), color='blue', alpha=0.6,
                     label='1 ' + son1.name)
    plot2 = plt.hist(son2.signal.fft_bins(), octave_histogram(fraction), color='orange', alpha=0.6,
                     label='2 ' + son2.name)
    plt.xscale('log')
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel('Amplitude')
    plt.grid('on')
    plt.legend()

    diff = plot1[0] - plot2[0]
    n_index = np.where(diff <= 0)[0]
    p_index = np.where(diff >= 0)[0]

    plt.subplot(1, 2, 2)
    # Negative difference corresponding to sound 2
    plt.bar(x_values[n_index], diff[n_index], width=bar_widths[n_index], color='orange', alpha=0.6)
    # Positive difference corresponding to sound1
    plt.bar(x_values[p_index], diff[p_index], width=bar_widths[p_index], color='blue', alpha=0.6)
    plt.title('Différence Son 1 - Son 2')
    plt.xscale('log')
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel('<- Son 2 : Son 1 ->')
    plt.grid('on')

"""
Classes
"""

class SP(object):
    """
    Sound parameters for the different analyses
    """

    class env(object):
        frame_size = 524  # frame size for the regular envelop in samples
        hop_length = None  # Default is 1/2 times the frame size

    class gen(object):
        octave_fraction = 3  # octave fraction for computations
        fft_range = 2000  # range of the FT plot in hz
        onset_delay = 100  # time in milliseconds to keep before the onset

    class log_e(object):
        """ Parameters for the log envelop method"""
        start_time = 0.01  # first time value for the log envelop plot
        min_window = None  # Default value is computed from the start time
        max_window = 2048  # Maximum width of the window

    class fund(object):
        """ Parameters for the function finding the fundamental"""
        min_freq = 60
        max_freq = 2000
        frame_length = 1024

    # Subclasses to store specific function parameters
    general = gen
    log_envelop = log_e
    fund = fund
    envelop = env
    # Frequency bins to divide the signal
    bins = {"bass": 100, "mid": 700, "highmid": 2000, "uppermid": 4000, "presence": 6000}

    def build_dict(self):
        pass

    def print_parameters(self):
        pass

class Signal(object):
    """
    Signal class to do computation on a audio signal the class tries to never change the .signal attribute
    """

    def __init__(self, signal, sr, SoundParam, range=None):
        """ Create a Signal class from a vector of samples and a sample rate"""
        self.SP = SoundParam
        self.onset = None
        self.signal = signal
        self.sr = sr
        self.get_envelop()
        self.time()
        self.get_log_envelop()
        self.get_fft()
        self.range = range
        self.trimmed = None

    def listen(self):
        """Method to listen the sound signal in a Jupyter Notebook"""
        file = 'temp.wav'
        write(file, self.signal, self.sr)
        ipd.display(ipd.Audio(file))
        os.remove(file)

    def plot(self, kind, **kwargs):
        """
        General plotting method for signals
        supported kinds of plots:
        - 'signal'
        - 'envelop'
        - 'norm_envelop'
        - 'fft'
        - 'spectrogram'
        """
        if kind == 'signal':

            plt.plot(self.t, self.signal, alpha=0.6, **kwargs)
            plt.xlabel('time (s)')
            plt.ylabel('amplitude')

        elif kind == 'envelop':

            plt.plot(self.envelop_time, self.envelop, **kwargs)
            plt.xlabel("time (s)")
            plt.ylabel("amplitude")

        elif kind == 'log envelop':
            plt.plot(self.log_envelop_time, self.log_envelop, **kwargs)
            plt.xlabel("time (s)")
            plt.ylabel("amplitude")

        elif kind == 'fft':
            # find the index corresponding to the fft range
            result = np.where(self.fft_freqs >= self.SP.general.fft_range)[0]
            if len(result) == 0:
                last_index = -1
            else:
                last_index = result[0]
            plt.plot(self.fft_freqs[:last_index], self.fft[:last_index], **kwargs)
            plt.xlabel("frequency"),
            plt.ylabel("amplitude"),
            plt.yscale('log')

        elif kind == 'fft hist':
            # Histogram of frequency values occurences in octave bins
            plt.hist(self.fft_bins(), octave_histogram(self.SP.general.octave_fraction), alpha=0.7, **kwargs)
            plt.xlabel('Fréquence (Hz)')
            plt.ylabel('Amplitude')
            plt.xscale('log')
            plt.yscale('log')
            plt.grid('on')

        elif kind == 'spectrogram':
            # Compute the spectrogram data
            D = librosa.stft(self.signal)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

            # Plot the spectrogram
            librosa.display.specshow(S_db, x_axis='time', y_axis='linear')

    def time(self):
        """ Method to create the time vector associated to the signal"""
        self.t = np.arange(0, len(self.signal) * (1 / self.sr), 1 / self.sr)

    def get_fft(self):
        """ Method to compute the fast fourrier transform of the signal"""
        fft = np.fft.fft(self.signal)
        self.fft = np.abs(fft[:int(len(fft) // 2)])  # Only the symmetric of the absolute value
        self.fft = self.fft / np.max(self.fft)
        fft_freqs = np.fft.fftfreq(len(fft), 1 / self.sr)  # Frequencies corresponding to the bins
        self.fft_freqs = fft_freqs[:int(len(fft) // 2)]

    def fft_bins(self):
        """ Method to get binned frequency values from the Fourier transform"""

        # Make the FT values integers
        fft_integers = [int(np.around(sample * 100, 0)) for sample in self.fft]

        # Create a list of the frequency occurences in the signal
        occurences = []
        for freq, count in zip(self.fft_freqs, fft_integers):
            occurences.append([freq] * count)

        # flatten the list
        return [item for sublist in occurences for item in sublist]

    def get_envelop(self):
        """
        Method calculating the amplitude envelope of a signal.
        frame_size : number of samples in the moving maximum
        hop_length : intersection between two successive frames
        """
        hop_length = self.SP.envelop.hop_length

        if hop_length is None:
            hop_length = int(self.SP.envelop.frame_size // 2)

        self.envelop = np.array(
            [max(self.signal[i:i + self.SP.envelop.frame_size]) for i in range(0, len(self.signal), hop_length)])
        frames = range(len(self.envelop))
        self.envelop_time = librosa.frames_to_time(frames, hop_length=hop_length)

    def get_log_envelop(self):
        """
        Method to compute a varying frame envelop to make a smooth plot on a log x scale
        start_time = minimum time value on plot
        min_window = minimum window length, if is None, is calculated from the start time to provide a smooth graph
        max_window = maximum window length, the maximum length of the sampling window of the envelop
        """

        if self.onset is None:
            self.onset = np.argmax(self.signal)
        else:
            self.onset = self.onset

        start_time = self.SP.log_envelop.start_time
        while start_time > (self.onset/self.sr):
            start_time = start_time / 10.

        start_exponent = int(np.log10(start_time))  # closest 10^x value for smooth graph

        if self.SP.log_envelop.min_window is None:
            min_window = 10 ** (start_exponent + 4)
            if min_window < 10:  # Value should at least be 10
                min_window = 10
        else:
            min_window = self.SP.log_envelop.min_window

        # initial values
        current_exponent = start_exponent
        current_time = 10 ** current_exponent  # start time on log scale
        index = int(current_time * self.sr)  # Start at the specified time
        window = min_window  # number of samples per window
        overlap = window // 2
        self.log_envelop = []
        self.log_envelop_time = [0]  # First value for comparison

        while index + window <= len(self.signal):

            while self.log_envelop_time[-1] < 11 ** (current_exponent + 1):
                if (index + window) < len(self.signal):
                    self.log_envelop.append(np.max(self.signal[index:index + window]))
                    self.log_envelop_time.append(self.t[index])
                    index += overlap
                else:
                    break

            if window * 10 < self.SP.log_envelop.max_window:
                window = window * 10
            else:
                window = self.SP.log_envelop.max_window

            overlap = window // 2
            current_exponent += 1

        # remove the value where t=0 so the log scale does not break
        self.log_envelop_time.remove(0)

    def trim_onset(self):
        """
        Trim the signal at the onset (max) minus the delay in miliseconds
        Return a trimmed signal with a noise attribute
        """
        delay_samples = int((self.SP.general.onset_delay / 1000) * self.sr)  # nb of samples to keep before the onset
        onset = np.argmax(self.signal)  # find the onset

        if onset > delay_samples:  # To make sure the index is positive
            trimmed_signal = Signal(self.signal[onset - delay_samples:], self.sr, self.SP)
            trimmed_signal.noise = self.signal[:onset - delay_samples]
            onset = np.argmax(trimmed_signal.envelop)
            trimmed_signal.onset = (trimmed_signal.envelop_time[onset], trimmed_signal.envelop[onset])
            trimmed_signal.trimmed = True
            trimmed_signal.onset = np.argmax(trimmed_signal.signal)
            return trimmed_signal

        else:
            print('Signal is too short to be trimmed before onset.')
            print('')
            self.trimmed = False
            return self

    def filter_noise(self):
        """ Method filtering the noise from the recorded signal and returning a filtered signal.
            If the signal was not trimmed it is trimmed in place then filtered.
            If the signal can not be trimmed it can't be filtered and the original signal is returned"""
        try:
            return Signal(reduce_noise(audio_clip=self.signal, noise_clip=self.noise), self.sr, self.SP)
        except AttributeError:
            if self.trimmed is False:
                print('Not sufficient noise in the raw signal, unable to filter.')
                print('')
                return self

    def normalise(self):
        """
        A function to normalise the signal to [-1,1]
        """
        return Signal(self.signal / np.max(np.abs(self.signal)), self.sr, self.SP)

    def make_freq_bins(self):
        """
        Method to divide a signal in frequency bins using butterworth filters
        bins are passed as a dict, default values are :
        - bass < 100 Hz
        - mid = 500
        - highmid = 2000 Hz
        - uppermid = 4000 Hz
        - presence > 6000 Hz
         The method return a dict with the divided signal as values and bin names as keys
        """

        bins = self.SP.bins

        bass_filter = sig.butter(12, bins["bass"], 'lp', fs=self.sr, output='sos')
        mid_filter = sig.butter(12, [bins["bass"], bins['mid']], 'bp', fs=self.sr, output='sos')
        himid_filter = sig.butter(12, [bins["mid"], bins['highmid']], 'bp', fs=self.sr, output='sos')
        upmid_filter = sig.butter(12, [bins["highmid"], bins['uppermid']], 'bp', fs=self.sr, output='sos')
        pres_filter = sig.butter(12, [bins["uppermid"], bins['presence']], 'bp', fs=self.sr, output='sos')
        bril_filter = sig.butter(12, bins['presence'], 'hp', fs=self.sr, output='sos')

        return {
            "bass": Signal(sig.sosfilt(bass_filter, self.signal), self.sr, self.SP, range=[0, bins["bass"]]),
            "mid": Signal(sig.sosfilt(mid_filter, self.signal), self.sr, self.SP, range=[bins["bass"], bins["mid"]]),
            "highmid": Signal(sig.sosfilt(himid_filter, self.signal), self.sr, self.SP, range=[bins["mid"], bins["highmid"]]),
            "uppermid": Signal(sig.sosfilt(upmid_filter, self.signal), self.sr, self.SP,
                               range=[bins["highmid"], bins["uppermid"]]),
            "presence": Signal(sig.sosfilt(pres_filter, self.signal), self.sr, self.SP,
                               range=[bins['uppermid'], bins["presence"]]),
            "brillance": Signal(sig.sosfilt(bril_filter, self.signal), self.sr, self.SP,
                                range=[bins["presence"], max(self.fft_freqs)])}

    def make_soundfile(self, name, path=''):
        """ Create a soundfile from a signal """
        write(path + name + ".wav", self.signal, self.sr)

class Sound(object):
    """A class to store audio signals obtained from a sound and compare them"""

    def __init__(self, file, name='', fundamental=None, SoundParams=None):
        """.__init__ method creating a Sound object from a .wav file, using Signal objects
            File loading returns a mono file if the file read is stereo
        """
        # Load the soundfile using librosa
        signal, sr = librosa.load(file)
        self.file = file

        # create a copy of the parameters
        if SoundParams is None:
            self.SP = SP()

        # create a Signal class from the signal and sample rate
        self.raw_signal = Signal(signal, sr, self.SP)

        # Allow user specified fundamental
        self.fundamental = fundamental
        self.name = name

    def change_params(self, SoundParams):
        self.__init__(self.file, name=self.name, fundamental=self.fundamental, SoundParams=SoundParams)
        self.condition()

    def condition(self):
        """ a general method applying all the pre-conditioning methods to the sound"""
        self.trim_signal()
        self.filter_noise()
        self.get_fundamental()
        self.bin_divide()

    def bin_divide(self):
        """ a method to divide the main signal into frequency bins"""
        # divide in frequency bins
        self.bins = self.signal.make_freq_bins()
        # unpack the bins
        self.bass, self.mid, self.highmid, self.uppermid, self.presence, self.brillance = self.bins.values()

    def filter_noise(self):
        """ a method to filter the noise from the trimmed signal"""
        # filter the noise in the Signal class
        self.signal = self.trimmed_signal.filter_noise()

    def trim_signal(self):
        """ a method to trim the signal to a specific delay before the onset, the default value is 100 ms"""
        # Trim the signal in the signal class
        self.trimmed_signal = self.raw_signal.trim_onset()

    def get_fundamental(self):
        """ finds the fundamental of the signal using a librosa function `librosa.yin`
            if the user specified a sound when instanciating the Sound class, this
            fundamental is used instead."""
        if self.fundamental is None:  # fundamental is not user specified
            self.fundamental = np.min(librosa.yin(self.signal.signal, self.SP.fund.min_freq, self.SP.fund.max_freq,
                                                  frame_length=self.SP.fund.frame_length))

    def validate_trim(self):
        """
        A function to perform a graphic validation of the transformation made by the function 'trim_onset()'
        """
        fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 6))
        ax1.plot(self.raw_signal.envelop_time, self.raw_signal.envelop, color='k')
        ax1.set(title='Old Envelop', xlabel='time', ylabel='amplitude')
        ax2.plot(self.trimmed_signal.envelop_time, self.trimmed_signal.envelop, color='k')
        ax2.plot(*self.trimmed_signal.onset, 'ro')
        ax2.set(title='Trimmed signal', xlabel='time', ylabel='amplitude')
        plt.tight_layout()

    def validate_noise(self):
        """Method to validate the noise filtering"""
        print('not filtered')
        self.trimmed_signal.listen()
        print('filtered')
        self.signal.listen()

    def listen_freq_bins(self):
        """ Method to listen to all the frequency bins of a sound"""
        for key in self.bins.keys():
            print(key)
            self.bins[key].listen()

    def plot_freq_bins(self):
        """ Method to plot all the frequency bins of a sound"""
        for key in self.bins.keys():
            lab = key + ' : ' + str(int(self.bins[key].range[0])) + ' - ' + str(int(self.bins[key].range[1])) + ' Hz'
            self.bins[key].plot('log envelop', label=lab)
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
