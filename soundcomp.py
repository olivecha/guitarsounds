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

def power_split(signal, x,  xmax, nbins):
    """ Return the index of the splitted bins of a signal with equal integrals"""
    imax = np.where(x > xmax)[0][0]
    A = scipy.integrate.trapezoid(signal[:imax])
    I = A / nbins
    indexes = [0]*nbins
    for i in range(nbins-1):
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
                son.bins[key].normalise().plot('envelop', label=(key + ' ' + str(i + 1)))
                plt.xscale('log')
                plt.legend()
    elif fbin in sons[0].bins.keys():
        plt.figure(figsize=(10, 8))
        # plot chaque son dans l'appel de la fonction
        for i, son in enumerate(sons):
            son.bins[fbin].normalise().plot('envelop', label=(fbin + ' ' + str(i + 1)))
            plt.xscale('log')
            plt.legend()
    else:
        print('fbin invalid')

def fft_mirror(son1, son2, max_freq=4000):
    index = np.where(son1.signal.fft_freqs > max_freq)[0][0]
    plt.figure(figsize=(10, 8))
    plt.yscale('symlog')
    plt.plot(son1.signal.fft_freqs[:index], son1.signal.fft[:index], label='1 : ' + son1.name)
    plt.plot(son2.signal.fft_freqs[:index], -son2.signal.fft[:index], label='2 : ' + son2.name)
    plt.legend()
    plt.show()

"""
Classes
"""


class Signal(object):
    """
    Signal class to do computation on a audio signal the class tries to never change the .signal attribute
    """

    def __init__(self, signal, sr, range=None):
        """ Create a Signal class from a vector of samples and a sample rate"""
        self.onset = None
        self.signal = signal
        self.sr = sr
        self.envelop()
        self.time()
        self.fft()
        self.range = range

    def listen(self):
        """Method to listen the sound signal in a Jupyter Notebook"""
        file = 'temp.wav'
        write(file, self.signal, self.sr)
        ipd.display(ipd.Audio(file))
        os.remove(file)

    def plot(self, kind, fft_range=2000, **kwargs):
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

        elif kind == 'fft':
            # find the index corresponding to the fft range
            result = np.where(self.fft_freqs >= fft_range)[0]
            if len(result) == 0:
                last_index = -1
            else:
                last_index = result[0]
            plt.plot(self.fft_freqs[:last_index], self.fft[:last_index], **kwargs)
            plt.xlabel("frequency"),
            plt.ylabel("amplitude"),
            plt.yscale('log')

        elif kind == 'spectrogram':
            # Compute the spectrogram data
            D = librosa.stft(self.signal)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

            # Plot the spectrogram
            librosa.display.specshow(S_db, x_axis='time', y_axis='linear')

    def time(self):
        """ Method to create the time vector associated to the signal"""
        self.t = np.arange(0, len(self.signal) * (1 / self.sr), 1 / self.sr)

    def fft(self):
        """ Method to compute the fast fourrier transform of the signal"""
        fft = np.fft.fft(self.signal)
        self.fft = np.abs(fft[:int(len(fft) // 2)])  # Only half of the absolute value
        fft_freqs = np.fft.fftfreq(len(fft), 1 / self.sr)  # Frequencies corresponding to the bins
        self.fft_freqs = fft_freqs[:int(len(fft) // 2)]

    def envelop(self, frame_size=512, hop_length=None):
        """
        Method calculating the amplitude envelope of a signal.
        frame_size : number of samples in the moving maximum
        hop_length : intersection between two successive frames
        """
        if hop_length is None:
            hop_length = int(frame_size // 2)

        if frame_size > hop_length:
            self.envelop = np.array(
                [max(self.signal[i:i + frame_size]) for i in range(0, len(self.signal), hop_length)])
            frames = range(len(self.envelop))
            self.envelop_time = librosa.frames_to_time(frames, hop_length=hop_length)
        else:
            print('hop_length must be less than frame size')

    def trim_onset(self, delay=100, inplace=False):
        """
        Trim the signal at the onset (max) minus the delay in miliseconds
        Return a trimmed signal with a noise attribute
        """
        delay_samples = int((delay / 1000) * self.sr)  # nb of samples to keep before the onset
        onset = np.argmax(self.signal)  # find the onset

        if onset > delay_samples:  # To make sure the index is positive
            if inplace is False:
                trimmed_signal = Signal(self.signal[onset - delay_samples:], self.sr)
                trimmed_signal.noise = self.signal[:onset - delay_samples]
                onset = np.argmax(trimmed_signal.envelop)
                trimmed_signal.onset = (trimmed_signal.envelop_time[onset], trimmed_signal.envelop[onset])
                return trimmed_signal
            else:
                # store the noise
                self.noise = self.signal[:onset - delay_samples]

                # trim the signal
                self.signal = self.signal[onset - delay_samples:]

                # Recompute the attributes
                self.time()
                self.envelop()
                self.fft()
        else:
            print('delay is too large')

    def filter_noise(self):
        """ Method filtering the noise from the recorded signal and returning a filtered signal"""
        try:
            return Signal(reduce_noise(audio_clip=self.signal, noise_clip=self.noise), self.sr)
        except AttributeError:
            self.trim_onset(inplace=True)
            return Signal(reduce_noise(audio_clip=self.signal, noise_clip=self.noise), self.sr)

    def normalise(self):
        return Signal(self.signal / np.max(self.signal), self.sr)

    def make_freq_bins(self, fundamental, bins=None, ):
        """
        Method to divide a signal in frequency bins using butterworth filters
        bins are passed as a dictionnary, default values are :
        - bass < 70 Hz
        - mid = 500
        - highmid = 2000 Hz
        - uppermid = 4000 Hz
        - presence > 6000 Hz
         The method return a dict with the divided signal as values and bin names as keys
        """

        if bins is None:
            # If the fundamental is between 0 and 150, it will be in the mid bin
            if fundamental < 200:
                bins = {"bass": fundamental, "mid": 700, "highmid": 2000, "uppermid": 4000, "presence": 6000}

        bass_filter = sig.butter(12, bins["bass"], 'lp', fs=self.sr, output='sos')
        mid_filter = sig.butter(12, [bins["bass"], bins['mid']], 'bp', fs=self.sr, output='sos')
        himid_filter = sig.butter(12, [bins["mid"], bins['highmid']], 'bp', fs=self.sr, output='sos')
        upmid_filter = sig.butter(12, [bins["highmid"], bins['uppermid']], 'bp', fs=self.sr, output='sos')
        pres_filter = sig.butter(12, [bins["uppermid"], bins['presence']], 'bp', fs=self.sr, output='sos')
        bril_filter = sig.butter(12, bins['presence'], 'hp', fs=self.sr, output='sos')

        return {
            "bass": Signal(sig.sosfilt(bass_filter, self.signal), self.sr, range=[0, bins["bass"]]),
            "mid": Signal(sig.sosfilt(mid_filter, self.signal), self.sr, range=[bins["bass"], bins["mid"]]),
            "highmid": Signal(sig.sosfilt(himid_filter, self.signal), self.sr, range=[bins["mid"], bins["highmid"]]),
            "uppermid": Signal(sig.sosfilt(upmid_filter, self.signal), self.sr, range=[bins["highmid"], bins["uppermid"]]),
            "presence": Signal(sig.sosfilt(pres_filter, self.signal), self.sr, range=[bins['uppermid'], bins["presence"]]),
            "brillance": Signal(sig.sosfilt(bril_filter, self.signal), self.sr, range=[bins["presence"], max(self.fft_freqs)])}

    def make_soundfile(self, name, path=''):
        """ Create a soundfile from a signal """
        write(path + name + ".wav", self.signal, self.sr)


class Sound(object):
    """A class to store audio signals obtained from a sound and compare them"""

    def __init__(self, file, name='', fundamental=None):
        """.__init__ method creating a Sound object from a .wav file, using Signal objects"""
        # Load the soundfile using librosa
        signal, sr = librosa.load(file)

        # create a Signal class from the signal and sample rate
        self.raw_signal = Signal(signal, sr)

        # Allow user specified fundamental
        self.fundamental = fundamental
        self.name = name

    def condition(self):
        """ a general method applying all the pre-conditioning methods to the sound"""
        self.trim_signal()
        self.filter_noise()
        self.get_fundamental()
        self.bin_divide()

    def bin_divide(self):
        """ a method to divide the main signal into frequency bins"""
        # divide in frequency bins
        self.bins = self.signal.make_freq_bins(self.fundamental)
        # unpack the bins
        self.bass, self.mid, self.highmid, self.uppermid, self.presence, self.brillance = self.bins.values()

    def filter_noise(self):
        """ a method to filter the noise from the trimmed signal"""
        # filter the noise in the Signal class
        self.signal = self.trimmed_signal.filter_noise()

    def trim_signal(self, delay=100):
        """ a method to trim the signal to a specific delay before the onset, the default value is 100 ms"""
        # Trim the signal in the signal class
        self.trimmed_signal = self.raw_signal.trim_onset(delay=delay)

    def get_fundamental(self):
        """ finds the fundamental of the signal using a librosa function `librosa.yin`
            if the user specified a sound when instanciating the Sound class, this
            fundamental is used instead."""
        if self.fundamental is None:  # fundamental is not user specified
            self.fundamental = np.mean(librosa.yin(self.signal.signal, 40, 2000)[3:-3])

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
            lab = key + ': [' + str(int(self.bins[key].range[0])) + ', ' + str(int(self.bins[key].range[1])) + ']'
            self.bins[key].plot('envelop', label=lab)
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
