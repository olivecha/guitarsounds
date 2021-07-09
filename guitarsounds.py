import librosa
import librosa.display
from soundfile import write
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
import os
from noisereduce import reduce_noise
import scipy
import scipy.integrate
from scipy import signal as sig
from guitarsounds_parameters import sound_parameters

"""
Getting the sound parameters from the guitarsounds_parameters.py file
"""

SP = sound_parameters()

"""
Global functions
"""

def compare(signal1, signal2, attribute,  **kwargs):
    """
    Side by side comparison of an attribute of two signals
    Ex : compare(Sound1.signal, Sound2.signal, 'fft') plots the fft of the two signals side by side
    :param signal1: First signal to be compared
    :param signal2: Second signal to be compared
    :param attribute: Attribute for comparison supported values are :
    'signal', 'envelop', 'norm_envelop', 'fft', 'fft hist', 'spectrogram'
    :param kwargs: key words arguments to pass to the two plots
    :return: None
    """
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    signal1.plot(kind=attribute, **kwargs)
    plt.subplot(1, 2, 2)
    signal2.plot(kind=attribute, **kwargs)

def octave_values(fraction, min_freq=10, max_freq=20200):
    """
    Compute octave fraction bin center values from min_freq to max_freq.
    `soundcomp.octave_values(3)` returns the bins corresponding to 1/3 octave values
    from 10 Hz to 20200 Hz.
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
    """
    # get the octave bins
    octave_bins = octave_values(fraction, **kwargs)

    # Compute the histogram bins
    hist_bins = []
    for f_o in octave_bins:
        # Intersecting lower bounds
        hist_bins.append(f_o / 2 ** (1 / (2 * fraction)))
    # Last upper bound
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

    :param sons: Sounds class instance to be compared
    :param fbin: frequency bins to compare, Supported arguments are :
    'all', 'bass', 'mid', 'highmid', 'uppermid', 'presence', 'brillance'
    :return: None
    """

    if fbin == 'all':
        # Create one plot per bin
        for key in [*list(SP.bins.__dict__.keys())[1:], 'brillance']:
            plt.figure(figsize=(10, 8))
            # plot every sound for a frequency bin
            norm_factors = np.array([son.bins[key].normalise().norm_factor for son in sons])
            for i, son in enumerate(sons):
                lab = ' ' + key + ' : ' + str(int(son.bins[key].range[0])) + ' - ' + str(
                    int(son.bins[key].range[1])) + ' Hz'
                son.bins[key].normalise().plot('log envelop', label=(str(i + 1) + '. ' + son.name + lab))
            plt.xscale('log')
            plt.legend()
            title1 = 'Normalisation Factor 1 : ' + str(np.around(norm_factors[0], 0)) + 'x, '
            title2 = 'Normalisation Factor 2 : ' + str(np.around(norm_factors[1], 0)) + 'x'
            plt.title(title1 + title2)

    elif fbin in [SP.bins.__dict__.keys(), 'brillance']:
        plt.figure(figsize=(10, 8))
        # Plot every envelop for a single frequency bin
        norm_factors = np.array([son.bins[fbin].normalise().norm_factor for son in sons])
        for i, son in enumerate(sons):
            lab = ' ' + fbin + ' : ' + str(int(son.bins[fbin].range[0])) + ' - ' + str(
                int(son.bins[fbin].range[1])) + ' Hz'
            son.bins[fbin].normalise().plot('log envelop', label=(str(i + 1) + '. ' + son.name + lab))
        plt.xscale('log')
        plt.legend()
        title1 = 'Normalisation Factor 1 : ' + str(np.around(norm_factors[0], 0)) + 'x\n'
        title2 = 'Normalisation Factor 2 : ' + str(np.around(norm_factors[1], 0)) + 'x'
        plt.title(title1 + title2)

    else:
        print('invalid frequency bin')

def peak_compare(son1, son2):
    index1 = np.where(son1.signal.fft_frequencies() > son1.SP.general.fft_range.value)[0][0]
    index2 = np.where(son2.signal.fft_frequencies() > son2.SP.general.fft_range.value)[0][0]

    # Get the peak data from the sounds
    peaks1 = son1.signal.peaks()
    peaks2 = son2.signal.peaks()
    freq1 = son1.signal.fft_frequencies()[:index1]
    freq2 = son2.signal.fft_frequencies()[:index2]
    fft1 = son1.signal.fft()[:index1]
    fft2 = son2.signal.fft()[:index2]

    peak_distance1 = np.mean([freq1[peaks1[i]] - freq1[peaks1[i+1]] for i in range(len(peaks1) - 1)]) / 4
    peak_distance2 = np.mean([freq2[peaks2[i]] - freq2[peaks2[i+1]] for i in range(len(peaks2) - 1)]) / 4
    peak_distance = np.abs(np.mean([peak_distance1, peak_distance2]))

    # Align  the two peak vectors
    new_peaks1 = []
    new_peaks2 = []
    for peak1 in peaks1:
        for peak2 in peaks2:
            if np.abs(freq1[peak1] - freq2[peak2]) < peak_distance:
                new_peaks1.append(peak1)
                new_peaks2.append(peak2)
    new_peaks1 = np.unique(np.array(new_peaks1))
    new_peaks2 = np.unique(np.array(new_peaks2))

    different_peaks1 = []
    different_peaks2 = []
    difference_treshold = 0.5
    while len(different_peaks1) < 1:
        for peak1, peak2 in zip(new_peaks1, new_peaks2):
            if np.abs(fft1[peak1] - fft2[peak2]) > difference_treshold:
                different_peaks1.append(peak1)
                different_peaks2.append(peak2)
        difference_treshold -= 0.01

    # Plot the output
    plt.figure(figsize=(10, 8))
    plt.yscale('symlog', linthresh=10e-4)
    # Sound1
    plt.plot(freq1, fft1, color='#919191', label='son 1')
    plt.scatter(freq1[peaks1], fft1[peaks1], color='r', label='old peaks')
    plt.scatter(freq1[new_peaks1], fft1[new_peaks1], color='b', label='new peaks')
    plt.scatter(freq1[different_peaks1], fft1[different_peaks1], color='g', label='diff peaks')
    annotation_string = 'Peaks with ' + str(np.around(difference_treshold, 1)) + ' difference'
    plt.annotate(annotation_string, (freq1[different_peaks1] + peak_distance / 2, fft1[different_peaks1]))

    # Sound2
    plt.plot(freq2, -fft2, color='#3d3d3d', label='son 2')
    plt.scatter(freq2[peaks2], -fft2[peaks2], color='r')
    plt.scatter(freq2[new_peaks2], -fft2[new_peaks2], color='b')
    plt.scatter(freq2[different_peaks2], -fft2[different_peaks2], color='g')

    plt.title('Fourier Transform Peak Analysis')
    plt.legend()

def fft_mirror(son1, son2):
    """ Plot the fourier transforms of two signals on the y and -y axes to compare them"""
    index = np.where(son1.signal.fft_frequencies() > SP.general.fft_range.value)[0][0]
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

def coherence(son1, son2, **kwargs):
    f, C = sig.coherence(son1.signal.signal, son2.signal.signal, son1.signal.sr)
    plt.plot(f, C, color='k', **kwargs)
    plt.yscale('log')
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel('Coherence [0, 1]')
    if (son1.name == '') | (son2.name == ''):
        title = 'Cohérence entre les deux sons'
    else:
        title = 'Cohérence entre les sons ' + son1.name + ' et ' + son2.name
    plt.title(title)

def average_envelop_plot(*sounds, kind='signal', difference_factor=1, show_sounds=True, show_rejects=True, **kwargs):
    sample_number = np.min([len(s1.signal.log_envelop()[0]) for s1 in sounds])

    if kind == 'signal':
        log_envelops = np.stack([s1.signal.normalise().log_envelop()[0][:sample_number] for s1 in sounds])
    elif kind in SP.bins.__dict__.keys():
        log_envelops = np.stack([s1.bins[kind].normalise().log_envelop()[0][:sample_number] for s1 in sounds])
    else:
        print('Wrong kind')

    average_log_envelop = np.mean(log_envelops, axis=0)
    std = np.std(log_envelops, axis=0)
    means = np.tile(average_log_envelop, (len(sounds), 1))
    diffs = np.sum(np.abs(means - log_envelops), axis=1)
    diff = np.mean(diffs)*difference_factor

    good_sounds = np.array(sounds)[diffs < diff]
    rejected_sounds = np.array(sounds)[diffs > diff]
    average_log_envelop = np.mean(log_envelops[diffs < diff], axis=0)
    norm_factors = np.array([s1.signal.normalise().norm_factor for s1 in good_sounds])

    if kind == 'signal':
        if show_sounds:
            for s1 in good_sounds[:-1]:
                s1.signal.normalise().plot(kind='log envelop', alpha=0.2, color='k')
            sounds[-1].signal.normalise().plot(kind='log envelop', alpha=0.2, color='k', label='sounds')

        if show_rejects:
            if len(rejected_sounds) > 1:
                for s1 in rejected_sounds[:-1]:
                    s1.signal.normalise().plot(kind='log envelop', alpha=0.3, color='r')
                rejected_sounds[-1].signal.normalise().plot(kind='log envelop', alpha=0.3, color='r',
                                                            label='rejected sounds')
            if len(rejected_sounds) == 1:
                rejected_sounds[0].signal.normalise().plot(kind='log envelop', alpha=0.3, color='r',
                                                           label='rejected sounds')
        if len(good_sounds) > 0:
            plt.plot(good_sounds[0].signal.log_envelop()[1], average_log_envelop, **kwargs)

    else:
        if show_sounds:
            for s1 in good_sounds[:-1]:
                s1.bins[kind].normalise().plot(kind='log envelop', alpha=0.2, color='k')
            sounds[-1].bins[kind].normalise().plot(kind='log envelop', alpha=0.2, color='k', label='sounds')

        if show_rejects:
            if len(rejected_sounds) > 1:
                for s2 in rejected_sounds[:-1]:
                    s1.bins[kind].normalise().plot(kind='log envelop', alpha=0.3, color='r')
                rejected_sounds[-1].bins[kind].normalise().plot(kind='log envelop', alpha=0.3, color='r',
                                                                label='rejected sounds')
            if len(rejected_sounds) == 1:
                rejected_sounds.bins[kind].normalise().plot(kind='log envelop', alpha=0.3, color='r',
                                                            label='rejected sounds')

        plt.plot(good_sounds[0].signal.log_envelop()[1][:sample_number], average_log_envelop, **kwargs)

    plt.xlabel('time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.xscale('log')
    print('Number of rejected sounds : ' + str(len(rejected_sounds)))
    print('Number of sounds included : ' + str(len(good_sounds)))
    print('Maximum normalisation factor : ' + str(np.around(np.max(norm_factors), 0)) + 'x')
    print('Minimum normalisation factor : ' + str(np.around(np.min(norm_factors), 0)) + 'x')

"""
Classes
"""

class Signal(object):
    """
    Signal class to do computation on a audio signal the class tries to never change the .signal attribute
    """

    def __init__(self, signal, sr, SoundParam, freq_range=None):
        """ Create a Signal class from a vector of samples and a sample rate"""
        self.SP = SoundParam
        self.onset = None
        self.signal = signal
        self.sr = sr
        self.range = freq_range
        self.trimmed = None

    def time(self):
        """
        Returns the time vector associated to the signal
        :return: numpy array corresponding to the time values of the signal samples in seconds
        """
        return np.arange(0, len(self.signal) * (1 / self.sr), 1 / self.sr)

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
        - 'fft hist'
        - 'spectrogram'
        """
        if kind == 'signal':

            plt.plot(self.time(), self.signal, alpha=0.6, **kwargs)
            plt.xlabel('time (s)')
            plt.ylabel('amplitude')

        elif kind == 'envelop':

            plt.plot(self.envelop_time(), self.envelop(), **kwargs)
            plt.xlabel("time (s)")
            plt.ylabel("amplitude")

        elif kind == 'log envelop':
            log_envelop, log_envelop_time = self.log_envelop()
            plt.plot(log_envelop_time, log_envelop, **kwargs)
            plt.xlabel("time (s)")
            plt.ylabel("amplitude")
            plt.xscale('log')

        elif kind == 'fft':
            # find the index corresponding to the fft range
            result = np.where(self.fft_frequencies() >= self.SP.general.fft_range.value)[0]
            if len(result) == 0:
                last_index = -1
            else:
                last_index = result[0]
            plt.plot(self.fft_frequencies()[:last_index], self.fft()[:last_index], **kwargs)
            plt.xlabel("frequency"),
            plt.ylabel("amplitude"),
            plt.yscale('log')

        elif kind == 'fft hist':
            # Histogram of frequency values occurences in octave bins
            plt.hist(self.fft_bins(), octave_histogram(self.SP.general.octave_fraction.value), alpha=0.7, **kwargs)
            plt.xlabel('Fréquence (Hz)')
            plt.ylabel('Amplitude')
            plt.xscale('log')
            plt.yscale('log')
            plt.grid('on')

        elif kind == 'spectrogram':
            # Spectrogram display from Librosa
            librosa.display.specshow(self.spectrogram(), x_axis='time', y_axis='linear')

    def spectrogram(self):
        # Compute the spectrogram data
        D = librosa.stft(self.signal)
        return librosa.amplitude_to_db(np.abs(D), ref=np.max)

    def fft(self):
        """
        Computes the Fast Fourier Transform of the signal and returns the vector.
        :return: Fast Fourier Transform amplitude values in a numpy array
        """
        fft = np.fft.fft(self.signal)
        fft = np.abs(fft[:int(len(fft) // 2)])  # Only the symmetric of the absolute value
        return fft / np.max(fft)

    def peaks(self):

        # Get the fft and fft frequencies from the signal
        fft, fft_freq = self.fft(), self.fft_frequencies()

        # Find the max index
        max_index = np.where(fft_freq >= self.SP.general.fft_range.value)[0][0]

        # Find an approximation of the distance between peaks, this only works for harmonic signals
        peak_distance = np.argmax(fft) // 2

        # Maximum of the signal in a small region on both ends
        fft_max_start = np.max(fft[:peak_distance])
        fft_max_end = np.max(fft[max_index - peak_distance:max_index])

        # Build the curve below the peaks but above the noise
        exponents = np.linspace(np.log10(fft_max_start), np.log10(fft_max_end), max_index)
        intersect = 10 ** exponents[peak_distance]
        diff_start = fft_max_start - intersect  # offset by a small distance so that the first max is not a peak
        min_height = 10 ** np.linspace(np.log10(fft_max_start + diff_start), np.log10(fft_max_end), max_index)

        first_peak_indexes, _ = sig.find_peaks(fft[:max_index], height=min_height, distance=peak_distance)

        number_of_peaks = len(first_peak_indexes)
        average_len = int(max_index / number_of_peaks) * 3

        if average_len % 2 == 0:
            average_len += 1

        average_fft = sig.savgol_filter(fft[:max_index], average_len, 1, mode='mirror') * 2

        peak_indexes, _ = sig.find_peaks(fft[:max_index], height=average_fft, distance=peak_distance)
        return peak_indexes

    def fft_frequencies(self):
        fft = self.fft()
        fft_frequencies = np.fft.fftfreq(len(fft), 1 / self.sr)  # Frequencies corresponding to the bins
        return fft_frequencies[:int(len(fft) // 2)]

    def fft_bins(self):
        """
        Method to get binned frequency values from the Fourier transform
        """

        # Make the FT values integers
        fft_integers = [int(np.around(sample * 100, 0)) for sample in self.fft()]

        # Create a list of the frequency occurrences in the signal
        occurrences = []
        for freq, count in zip(self.fft_frequencies(), fft_integers):
            occurrences.append([freq] * count)

        # flatten the list
        return [item for sublist in occurrences for item in sublist]

    def envelop(self):
        """
        Method calculating the amplitude envelope of a signal.
        :return: Amplitude envelop of the signal
        """
        # Get the hop length
        hop_length = self.SP.envelop.hop_length.value

        # Compute the envelop
        envelop = np.array(
            [max(self.signal[i:i + self.SP.envelop.frame_size.value]) for i in range(0, len(self.signal), hop_length)])

        return envelop

    def envelop_time(self):
        """
        Method calculating the time vector associated to a signal envelop
        :return: Time vector associated to the signal envelop
        """
        # Get the number of frames from the signal envelop
        frames = range(len(self.envelop()))
        # Return the envelop frames computed with Librosa
        return librosa.frames_to_time(frames, hop_length=self.SP.envelop.hop_length.value)

    def log_envelop(self):
        if self.onset is None:
            onset = np.argmax(self.signal)
        else:
            onset = self.onset

        start_time = self.SP.log_envelop.start_time.value
        while start_time > (onset / self.sr):
            start_time /= 10.

        start_exponent = int(np.log10(start_time))  # closest 10^x value for smooth graph

        if self.SP.log_envelop.min_window.value is None:
            min_window = 15 ** (start_exponent + 4)
            if min_window < 15:  # Value should at least be 10
                min_window = 15
        else:
            min_window = self.SP.log_envelop.min_window.value

        # initial values
        current_exponent = start_exponent
        current_time = 10 ** current_exponent  # start time on log scale
        index = int(current_time * self.sr)  # Start at the specified time
        window = min_window  # number of samples per window
        overlap = window // 2
        log_envelop = []
        log_envelop_time = [0]  # First value for comparison

        while index + window <= len(self.signal):

            while log_envelop_time[-1] < 10 ** (current_exponent + 1):
                if (index + window) < len(self.signal):
                    log_envelop.append(np.max(self.signal[index:index + window]))
                    log_envelop_time.append(self.time()[index])
                    index += overlap
                else:
                    break

            if window * 10 < self.SP.log_envelop.max_window.value:
                window = window * 10
            else:
                window = self.SP.log_envelop.max_window.value

            overlap = window // 2
            current_exponent += 1

        # remove the value where t=0 so the log scale does not break
        log_envelop_time.remove(0)

        return log_envelop, log_envelop_time

    def trim_onset(self, verbose=True):
        """
        Trim the signal at the onset (max) minus the delay in miliseconds
        Return a trimmed signal with a noise attribute
        """
        # nb of samples to keep before the onset
        delay_samples = int((self.SP.general.onset_delay.value / 1000) * self.sr)
        onset = np.argmax(self.signal)  # find the onset

        if onset > delay_samples:  # To make sure the index is positive
            trimmed_signal = Signal(self.signal[onset - delay_samples:], self.sr, self.SP)
            trimmed_signal.noise = self.signal[:onset - delay_samples]
            onset = np.argmax(trimmed_signal.envelop())
            trimmed_signal.onset = (trimmed_signal.envelop_time()[onset], trimmed_signal.envelop()[onset])
            trimmed_signal.trimmed = True
            trimmed_signal.onset = np.argmax(trimmed_signal.signal)
            return trimmed_signal

        else:
            if verbose:
                print('Signal is too short to be trimmed before onset.')
                print('')
            self.trimmed = False
            return self

    def filter_noise(self, verbose=True):
        """ Method filtering the noise from the recorded signal and returning a filtered signal.
            If the signal was not trimmed it is trimmed in place then filtered.
            If the signal can not be trimmed it can't be filtered and the original signal is returned"""
        try:
            return Signal(reduce_noise(audio_clip=self.signal, noise_clip=self.noise), self.sr, self.SP)
        except AttributeError:
            if self.trimmed is False:
                if verbose:
                    print('Not sufficient noise in the raw signal, unable to filter.')
                    print('')
                return self

    def normalise(self):
        """
        A function to normalise the signal to [-1,1]
        """
        normalised_signal = Signal(self.signal / np.max(np.abs(self.signal)), self.sr, self.SP)
        normalised_signal.norm_factor = (1/np.max(np.abs(self.signal)))
        return normalised_signal

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

        bins = self.SP.bins.__dict__

        bass_filter = sig.butter(12, bins["bass"].value, 'lp', fs=self.sr, output='sos')
        mid_filter = sig.butter(12, [bins["bass"].value, bins['mid'].value], 'bp', fs=self.sr, output='sos')
        himid_filter = sig.butter(12, [bins["mid"].value, bins['highmid'].value], 'bp', fs=self.sr, output='sos')
        upmid_filter = sig.butter(12, [bins["highmid"].value, bins['uppermid'].value], 'bp', fs=self.sr, output='sos')
        pres_filter = sig.butter(12, [bins["uppermid"].value, bins['presence'].value], 'bp', fs=self.sr, output='sos')
        bril_filter = sig.butter(12, bins['presence'].value, 'hp', fs=self.sr, output='sos')

        return {
            "bass": Signal(sig.sosfilt(bass_filter, self.signal), self.sr, self.SP,
                           freq_range=[0, bins["bass"].value]),
            "mid": Signal(sig.sosfilt(mid_filter, self.signal), self.sr, self.SP,
                          freq_range=[bins["bass"].value, bins["mid"].value]),
            "highmid": Signal(sig.sosfilt(himid_filter, self.signal), self.sr, self.SP,
                              freq_range=[bins["mid"].value, bins["highmid"].value]),
            "uppermid": Signal(sig.sosfilt(upmid_filter, self.signal), self.sr, self.SP,
                               freq_range=[bins["highmid"].value, bins["uppermid"].value]),
            "presence": Signal(sig.sosfilt(pres_filter, self.signal), self.sr, self.SP,
                               freq_range=[bins['uppermid'].value, bins["presence"].value]),
            "brillance": Signal(sig.sosfilt(bril_filter, self.signal), self.sr, self.SP,
                                freq_range=[bins["presence"].value, max(self.fft_frequencies())])}

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
            self.SP = SP
        else:
            self.SP = SoundParams

        # create a Signal class from the signal and sample rate
        self.raw_signal = Signal(signal, sr, self.SP)

        # Allow user specified fundamental
        self.fundamental = fundamental
        self.name = name

    def condition(self, verbose=True):
        """ a general method applying all the pre-conditioning methods to the sound"""
        self.trim_signal(verbose=verbose)
        self.filter_noise(verbose=verbose)
        self.get_fundamental()
        self.bin_divide()

    def bin_divide(self):
        """ a method to divide the main signal into frequency bins"""
        # divide in frequency bins
        self.bins = self.signal.make_freq_bins()
        # unpack the bins
        self.bass, self.mid, self.highmid, self.uppermid, self.presence, self.brillance = self.bins.values()

    def filter_noise(self, verbose=True):
        """ a method to filter the noise from the trimmed signal"""
        # filter the noise in the Signal class
        self.signal = self.trimmed_signal.filter_noise(verbose=verbose)

    def trim_signal(self, verbose=True):
        """ a method to trim the signal to a specific delay before the onset, the default value is 100 ms"""
        # Trim the signal in the signal class
        self.trimmed_signal = self.raw_signal.trim_onset(verbose=verbose)

    def get_fundamental(self):
        """ finds the fundamental of the signal using a librosa function `librosa.yin`
            if the user specified a sound when instantiating the Sound class, this
            fundamental is used instead."""
        if self.fundamental is None:  # fundamental is not user specified
            self.fundamental = np.min(librosa.yin(self.signal.signal, self.SP.fundamental.min_freq.value,
                                                  self.SP.fundamental.max_freq.value,
                                                  frame_length=self.SP.fundamental.frame_length.value))

    def validate_trim(self):
        """
        A function to perform a graphic validation of the transformation made by the function 'trim_onset()'
        """
        fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 6))
        ax1.plot(self.raw_signal.envelop_time(), self.raw_signal.envelop(), color='k')
        ax1.set(title='Old Envelop', xlabel='time', ylabel='amplitude')
        ax2.plot(self.trimmed_signal.envelop_time(), self.trimmed_signal.envelop(), color='k')
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

    def plot_freq_bins(self, keys=None):
        """ Method to plot all the frequency bins of a sound"""
        if keys is None:
            keys = self.bins.keys()
        for key in keys:
            lab = key + ' : ' + str(int(self.bins[key].range[0])) + ' - ' + str(int(self.bins[key].range[1])) + ' Hz'
            self.bins[key].plot('log envelop', label=lab)
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
