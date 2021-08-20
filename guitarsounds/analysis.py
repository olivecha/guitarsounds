import librosa
import librosa.display
from soundfile import write
import IPython.display as ipd
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
import os
from noisereduce.noisereducev1 import reduce_noise
import scipy
import scipy.optimize
import scipy.integrate
import scipy.interpolate
from scipy import signal as sig
from guitarsounds.parameters import sound_parameters
import guitarsounds.utils as utils
from tabulate import tabulate

"""
Getting the sound parameters from the guitarsounds_parameters.py file
"""
SP = sound_parameters()

"""
Classes
"""

class SoundPack(object):
    """
    A class to store and analyse multiple sounds
    Some methods are only available for the case with two sounds
    """

    def __init__(self, *sounds, names=None, fundamentals=None, SoundParams=None, equalize_time=True):
        """
        The SoundPack can be instantiated from existing Sound class instances, either in a list or as
        multiple arguments

        The class can also handle the creation of Sound class instances if the arguments are filenames,
        either a list or multiple arguments.

        If the number of Sound contained is equal to two, the SoundPack will be 'dual' and the associated methods
        will be available

        If it contains multiple sounds the SoundPack will be multiple and a reduced number of methods will work

        A list of names as strings and fundamental frequencies can be specified when creating the SoundPack

        If equalize_time is set to False, the contained sounds will not be trimmed to the same length.

        Examples :
        ```
        Sound_Test = SoundPack('sounds/test1.wav', 'sounds/test2.wav', names=['A', 'B'], fundamentals = [134, 134])

        sounds = [sound1, sound2, sound3, sound4, sound5] # instances of the Sound class
        large_Test = SoundPack(sounds, names=['1', '2', '3', '4', '5'])
        ```
        """
        # create a copy of the sound parameters
        if SoundParams is None:
            self.SP = SP
        else:
            self.SP = SoundParams

        # Check if the sounds argument is a list
        if type(sounds[0]) is list:
            sounds = sounds[0]  # unpack the list

        # Check for special case
        if len(sounds) == 2:
            # special case to compare two sounds
            self.kind = 'dual'

        elif len(sounds) > 1:
            # general case for multiple sounds
            self.kind = 'multiple'

        if type(sounds[0]) is str:
            self.sounds_from_files(sounds, names=names, fundamentals=fundamentals)

        else:
            self.sounds = sounds
            if names is None:
                names = [str(n) for n in np.arange(1, len(sounds)+1)]
                for sound, n in zip(self.sounds, names):
                    sound.name = n

        if equalize_time:
            self.equalize_time()

    def sounds_from_files(self, sound_files, names=None, fundamentals=None):
        """
        Create Sound class instances and assign them to the SoundPack from a list of files
        :param sound_files: sound filenames
        :param names: sound names
        :param fundamentals: user specified fundamental frequencies
        :return: None
        """
        # Make the default name list from sound filenames if none is supplied
        if (names is None) or (len(names) != len(sound_files)):
            names = [file[:-4] for file in sound_files]  # remove the .wav

        # If the fundamentals are not supplied or mismatch in number None is used
        if (fundamentals is None) or (len(fundamentals) != len(sound_files)):
            fundamentals = len(sound_files) * [None]

        # Create Sound instances from files
        self.sounds = []
        for file, name, fundamental in zip(sound_files, names, fundamentals):
            self.sounds.append(Sound(file, name=name, fundamental=fundamental,
                                     SoundParams=self.SP).condition(return_self=True))

    def equalize_time(self):
        """
        Trim the sounds so that they all have the length of the shortest sound, trimming is done at the end.
        :return: None
        """
        trim_index = np.min([len(sound.signal.signal) for sound in self.sounds])
        trimmed_sounds = []
        for sound in self.sounds:
            new_sound = sound
            new_sound.signal = new_sound.signal.trim_time(trim_index / sound.signal.sr)
            new_sound.bin_divide()
            trimmed_sounds.append(new_sound)
        self.sounds = trimmed_sounds

    def normalize(self):
        """
        Normalize all the signals in the SoundPack and returns a normaized
        instance of itself
        :return: SoundPack with normalized signals
        """
        new_sounds = []
        names = [sound.name for sound in self.sounds]
        fundamentals = [sound.fundamental for sound in self.sounds]
        for sound in self.sounds:
            sound.signal = sound.signal.normalize()
            new_sounds.append(sound)

        return SoundPack(new_sounds, names=names, fundamentals=fundamentals, SoundParams=self.SP, equalize_time=False)

    """
    Methods for all SoundPacks
    """

    def plot(self, kind, **kwargs):
        """
        __ Multiple SoundPack Method __
        Plots a specific signal.plot for all sounds on the same figure
        Ex : compare_plot('fft') plots the fft of all sounds on a single figure
        The color argument is set to none so that the plots have different colors

        :param kind: Attribute passed to the `signal.plot()` method
        :param kwargs: key words arguments to pass to the `signal.plot()` method
        :return: None
        """
        for sound in self.sounds:
            kwargs['label'] = sound.name
            kwargs['color'] = None
            sound.signal.plot(kind, **kwargs)

        plt.title(kind + ' plot')
        plt.legend()

    def compare_plot(self, kind, **kwargs):
        """
        __ Multiple SoundPack Method __
        Draws the same kind of plot on a different axis for each sound
        Example : `SoundPack.compare_plot('peaks')` with 4 Sounds will plot a figure with 4 axes, with each
        a different 'peak' plot.

        :param kind: kind argument passed to `Signal.plot()`
        :param kwargs: key word arguments passed to Signal.plot()
        :return: None
        """
        # if a dual SoundPack : only plot two big plots
        if self.kind == 'dual':
            fig, axs = plt.subplots(1, 2, figsize=(12, 4))
            for sound, ax in zip(self.sounds, axs):
                plt.sca(ax)
                sound.signal.plot(kind, **kwargs)
                ax.set_title(kind + ' ' + sound.name)

        # If a multiple SoundPack : plot on a grid of axes
        elif self.kind == 'multiple':

            # find the n, m values for the subplots line and columns
            n = len(self.sounds)
            if n // 4 >= 10:
                # a lot of sounds
                cols = 4
            elif n // 3 >= 10:
                # many sounds
                cols = 3
            elif n // 2 <= 4:
                # a few sounds
                cols = 2

            remainder = n % cols
            if remainder == 0:
                rows = n // cols
            else:
                rows = n // cols + 1

            fig, axs = plt.subplots(rows, cols, figsize=(12, 4*rows))
            axs = axs.reshape(-1)
            for sound, ax in zip(self.sounds, axs):
                plt.sca(ax)
                sound.signal.plot(kind, **kwargs)
                title = ax.get_title()
                title = sound.name + ' ' + title
                ax.set_title(title)

            if remainder != 0:
                for ax in axs[-(cols - remainder):]:
                    ax.set_axis_off()

            plt.tight_layout()

    def freq_bin_plot(self, fbin='all'):
        """
        __ Multiple SoundPack Method __
        A function to compare signals decomposed frequency wise in the time domain on a logarithm scale.
        The methods plots all the sounds and plots their frequency bins according to the frequency bin argument fbin.

        Example : SoundPack.freq_bin_plot(fbin='mid') will plot the log-scale envelop of the 'mid' signal of every
        sound in the SoundPack

        :param fbin: frequency bins to compare, Supported arguments are :
        'all', 'bass', 'mid', 'highmid', 'uppermid', 'presence', 'brillance'
        :return: None
        """

        if fbin == 'all':
            # Create one plot per bin
            for key in [*list(self.SP.bins.__dict__.keys())[1:], 'brillance']:
                plt.figure(figsize=(10, 4))
                # plot every sound for a frequency bin
                norm_factors = np.array([son.bins[key].normalize().norm_factor for son in self.sounds])
                for i, son in enumerate(self.sounds):
                    lab = ' ' + key + ' : ' + str(int(son.bins[key].range[0])) + ' - ' + str(
                        int(son.bins[key].range[1])) + ' Hz'
                    son.bins[key].normalize().plot('log envelop', label=(str(i + 1) + '. ' + son.name + lab))
                plt.xscale('log')
                plt.legend()
                title1 = 'Normalisation Factors : '
                title2 = 'x, '.join(str(np.around(norm_factor, 0)) for norm_factor in norm_factors)
                plt.title(title1 + title2)

        elif fbin in [*list(SP.bins.__dict__.keys())[1:], 'brillance']:
            plt.figure(figsize=(10, 4))
            # Plot every envelop for a single frequency bin
            norm_factors = np.array([son.bins[fbin].normalize().norm_factor for son in self.sounds])
            for i, son in enumerate(self.sounds):
                lab = ' ' + fbin + ' : ' + str(int(son.bins[fbin].range[0])) + ' - ' + str(
                    int(son.bins[fbin].range[1])) + ' Hz'
                son.bins[fbin].normalize().plot('log envelop', label=(str(i + 1) + '. ' + son.name + lab))
            plt.xscale('log')
            plt.legend()
            title1 = 'Normalisation Factor 1 : ' + str(np.around(norm_factors[0], 0)) + 'x\n'
            title2 = 'Normalisation Factor 2 : ' + str(np.around(norm_factors[1], 0)) + 'x'
            plt.title(title1 + title2)

        else:
            print('invalid frequency bin')

    def combine_envelop(self, kind='signal', difference_factor=1, show_sounds=True, show_rejects=True, **kwargs):
        """
        __ Multiple SoundPack Method __
        Combines the envelops of the Sounds contained in the SoundPack, Sounds having a too large difference factor
        from the average are rejected.

        :param kind: wich signal to use from :
        'signal', 'bass', 'mid', 'highmid', 'uppermid', 'presence', 'brillance'
        :param difference_factor: threshold to reject a sound from the combinaison, can be adjusted to reject
        or include more sounds.
        :param show_sounds: If True all the included Sounds are shown on the plot
        :param show_rejects: If True all the rejected Sounds are shown on the plot
        :param kwargs: Key word arguments to pass to the envelop plot.
        :return: None
        """
        sounds = self.sounds
        sample_number = np.min([len(s1.signal.log_envelop()[0]) for s1 in sounds])

        if kind == 'signal':
            log_envelops = np.stack([s1.signal.normalize().log_envelop()[0][:sample_number] for s1 in sounds])
        elif kind in SP.bins.__dict__.keys():
            log_envelops = np.stack([s1.bins[kind].normalize().log_envelop()[0][:sample_number] for s1 in sounds])
        else:
            print('Wrong kind')

        average_log_envelop = np.mean(log_envelops, axis=0)
        means = np.tile(average_log_envelop, (len(sounds), 1))
        diffs = np.sum(np.abs(means - log_envelops), axis=1)
        diff = np.mean(diffs) * difference_factor

        good_sounds = np.array(sounds)[diffs < diff]
        rejected_sounds = np.array(sounds)[diffs > diff]
        average_log_envelop = np.mean(log_envelops[diffs < diff], axis=0)
        norm_factors = np.array([s1.signal.normalize().norm_factor for s1 in good_sounds])

        if kind == 'signal':
            if show_sounds:
                for s1 in good_sounds[:-1]:
                    s1.signal.normalize().plot(kind='log envelop', alpha=0.2, color='k')
                sounds[-1].signal.normalize().plot(kind='log envelop', alpha=0.2, color='k', label='sounds')

            if show_rejects:
                if len(rejected_sounds) > 1:
                    for s1 in rejected_sounds[:-1]:
                        s1.signal.normalize().plot(kind='log envelop', alpha=0.3, color='r')
                    rejected_sounds[-1].signal.normalize().plot(kind='log envelop', alpha=0.3, color='r',
                                                                label='rejected sounds')
                if len(rejected_sounds) == 1:
                    rejected_sounds[0].signal.normalize().plot(kind='log envelop', alpha=0.3, color='r',
                                                               label='rejected sounds')
            if len(good_sounds) > 0:
                if 'label' in kwargs.keys():
                    plt.plot(good_sounds[0].signal.log_envelop()[1][:len(average_log_envelop)], average_log_envelop,
                             **kwargs)
                else:
                    plt.plot(good_sounds[0].signal.log_envelop()[1][:len(average_log_envelop)], average_log_envelop,
                             label='average', color='k', **kwargs)

        else:
            if show_sounds:
                for s1 in good_sounds[:-1]:
                    s1.bins[kind].normalize().plot(kind='log envelop', alpha=0.2, color='k')
                sounds[-1].bins[kind].normalize().plot(kind='log envelop', alpha=0.2, color='k', label='sounds')

            if show_rejects:
                if len(rejected_sounds) > 1:
                    for s2 in rejected_sounds[:-1]:
                        s2.bins[kind].normalize().plot(kind='log envelop', alpha=0.3, color='r')
                    rejected_sounds[-1].bins[kind].normalize().plot(kind='log envelop', alpha=0.3, color='r',
                                                                    label='rejected sounds')
                if len(rejected_sounds) == 1:
                    rejected_sounds.bins[kind].normalize().plot(kind='log envelop', alpha=0.3, color='r',
                                                                label='rejected sounds')

            plt.plot(good_sounds[0].signal.log_envelop()[1][:sample_number], average_log_envelop, color='k', **kwargs)

        plt.xlabel('time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.xscale('log')
        print('Number of rejected sounds : ' + str(len(rejected_sounds)))
        print('Number of sounds included : ' + str(len(good_sounds)))
        print('Maximum normalisation factor : ' + str(np.around(np.max(norm_factors), 0)) + 'x')
        print('Minimum normalisation factor : ' + str(np.around(np.min(norm_factors), 0)) + 'x')

    def fundamentals(self):
        """
        __ Multiple Soundpack Method __
        Displays the fundamentals of every sound in the SoundPack
        :return: None
        """
        names = np.array([sound.name for sound in self.sounds])
        fundamentals = np.array([np.around(sound.fundamental, 1) for sound in self.sounds])
        key = np.argsort(fundamentals)
        table_data = [names[key], fundamentals[key]]

        table_data = np.array(table_data).transpose()

        print(tabulate(table_data, headers=['Name', 'Fundamental (Hz)']))

    """
    Methods for dual SoundPacks
    """

    def compare_peaks(self):
        """
        __ Dual SoundPack Method __
        Compares the peaks in the Fourier Transform of two Sounds,
        the peak with the highest difference is highlighted
        :return: None
        """
        if self.kind == 'dual':
            son1 = self.sounds[0]
            son2 = self.sounds[1]
            index1 = np.where(son1.signal.fft_frequencies() > self.SP.general.fft_range.value)[0][0]
            index2 = np.where(son2.signal.fft_frequencies() > self.SP.general.fft_range.value)[0][0]

            # Get the peak data from the sounds
            peaks1 = son1.signal.peaks()
            peaks2 = son2.signal.peaks()
            freq1 = son1.signal.fft_frequencies()[:index1]
            freq2 = son2.signal.fft_frequencies()[:index2]
            fft1 = son1.signal.fft()[:index1]
            fft2 = son2.signal.fft()[:index2]

            peak_distance1 = np.mean([freq1[peaks1[i]] - freq1[peaks1[i + 1]] for i in range(len(peaks1) - 1)]) / 4
            peak_distance2 = np.mean([freq2[peaks2[i]] - freq2[peaks2[i + 1]] for i in range(len(peaks2) - 1)]) / 4
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
            difference_threshold = 0.5
            while len(different_peaks1) < 1:
                for peak1, peak2 in zip(new_peaks1, new_peaks2):
                    if np.abs(fft1[peak1] - fft2[peak2]) > difference_threshold:
                        different_peaks1.append(peak1)
                        different_peaks2.append(peak2)
                difference_threshold -= 0.01

            # Plot the output
            plt.figure(figsize=(10, 6))
            plt.yscale('symlog', linthresh=10e-1)

            # Sound 1
            plt.plot(freq1, fft1, color='#919191', label=son1.name)
            plt.scatter(freq1[new_peaks1], fft1[new_peaks1], color='b', label='peaks')
            plt.scatter(freq1[different_peaks1], fft1[different_peaks1], color='g', label='diff peaks')
            annotation_string = 'Peaks with ' + str(np.around(difference_threshold, 2)) + ' difference'
            plt.annotate(annotation_string, (freq1[different_peaks1[0]] + peak_distance / 2, fft1[different_peaks1[0]]))

            # Sound2
            plt.plot(freq2, -fft2, color='#3d3d3d', label=son2.name)
            plt.scatter(freq2[new_peaks2], -fft2[new_peaks2], color='b')
            plt.scatter(freq2[different_peaks2], -fft2[different_peaks2], color='g')
            plt.title('Fourier Transform Peak Analysis for ' + son1.name + ' and ' + son2.name)
            plt.legend()
        else:
            print('Unsupported for multiple sounds SoundPacks')

    def fft_mirror(self):
        """
        __ Dual SoundPack Method __
        Plot the fourier transforms of two sounds on the y and -y axes to compare them.
        :return: None
        """
        if self.kind == 'dual':
            son1 = self.sounds[0]
            son2 = self.sounds[1]
            index = np.where(son1.signal.fft_frequencies() > SP.general.fft_range.value)[0][0]

            plt.figure(figsize=(10, 6))
            plt.yscale('symlog')
            plt.grid('on')
            plt.plot(son1.signal.fft_frequencies()[:index], son1.signal.fft()[:index], label=son1.name)
            plt.plot(son2.signal.fft_frequencies()[:index], -son2.signal.fft()[:index], label=son2.name)
            plt.xlabel('Fréquence (Hz)')
            plt.ylabel('Amplitude')
            plt.title('Mirror Fourier Transform for ' + son1.name + ' and ' + son2.name)
            plt.legend()

        else:
            print('Unsupported for multiple sounds SoundPacks')

    def fft_diff(self, fraction=3, ticks=None):
        """
        __ Dual SoundPack Method __
        Compare the Fourier Transform of two sounds by computing the differences of the octave bins heights.
        The two FTs are superimposed on the first plot to show differences
        The difference between the two FTs is plotted on the second plot

        :param fraction: octave fraction value used to compute the frequency bins A higher number will show
        a more precise comparison, but conclusions may be harder to draw.
        :param ticks:  If True the frequency bins intervals are used as X axis ticks
        :return: None
        """
        if self.kind == 'dual':
            # Separate the sounds
            son1 = self.sounds[0]
            son2 = self.sounds[1]

            # Compute plotting bins
            x_values = utils.octave_values(fraction)
            hist_bins = utils.octave_histogram(fraction)
            bar_widths = np.array([hist_bins[i + 1] - hist_bins[i] for i in range(0, len(hist_bins) - 1)])

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            plot1 = ax1.hist(son1.signal.fft_bins(), utils.octave_histogram(fraction), color='blue', alpha=0.6,
                             label=son1.name)
            plot2 = ax1.hist(son2.signal.fft_bins(), utils.octave_histogram(fraction), color='orange', alpha=0.6,
                             label=son2.name)
            ax1.set_title('FT Histogram for ' + son1.name + ' and ' + son2.name)
            ax1.set_xscale('log')
            ax1.set_xlabel('Fréquence (Hz)')
            ax1.set_ylabel('Amplitude')
            ax1.grid('on')
            ax1.legend()

            diff = plot1[0] - plot2[0]
            n_index = np.where(diff <= 0)[0]
            p_index = np.where(diff >= 0)[0]

            # Negative difference corresponding to sound 2
            ax2.bar(x_values[n_index], diff[n_index], width=bar_widths[n_index], color='orange', alpha=0.6)
            # Positive difference corresponding to sound1
            ax2.bar(x_values[p_index], diff[p_index], width=bar_widths[p_index], color='blue', alpha=0.6)
            ax2.set_title('Difference ' + son1.name + ' - ' + son2.name)
            ax2.set_xscale('log')
            ax2.set_xlabel('Fréquence (Hz)')
            ax2.set_ylabel('<- Son 2 : Son 1 ->')
            ax2.grid('on')

            if ticks == 'bins':
                labels = [label for label in self.SP.bins.__dict__ if label != 'name']
                labels.append('brillance')
                x = [param.value for param in self.SP.bins.__dict__.values() if param != 'bins']
                x.append(11250)
                x_formatter = ticker.FixedFormatter(labels)
                x_locator = ticker.FixedLocator(x)
                ax1.xaxis.set_major_locator(x_locator)
                ax1.xaxis.set_major_formatter(x_formatter)
                ax1.tick_params(axis="x", labelrotation=90)
                ax2.xaxis.set_major_locator(x_locator)
                ax2.xaxis.set_major_formatter(x_formatter)
                ax2.tick_params(axis="x", labelrotation=90)

        else:
            print('Unsupported for multiple sounds SoundPacks')

    def coherence_plot(self):
        """
        __ Dual SoundPack Method __
        computes and plots the coherence between the time signal of two Sounds
        :return: None
        """
        if self.kind == 'dual':
            f, C = sig.coherence(self.sounds[0].signal.signal, self.sounds[1].signal.signal, self.sounds[0].signal.sr)
            plt.plot(f, C, color='b')
            plt.yscale('log')
            plt.xlabel('Fréquence (Hz)')
            plt.ylabel('Coherence [0, 1]')
            title = 'Cohérence entre les sons ' + self.sounds[0].name + ' et ' + self.sounds[1].name
            plt.title(title)
        else:
            print('Unsupported for multiple sounds SoundPacks')


class Signal(object):
    """
    A Class to do computations on an audio signal.

    The signal is never changed in the class, when transformations are made, a new instance is returned.

    """

    def __init__(self, signal, sr, SoundParam, freq_range=None):
        """ Create a Signal class from a vector of samples and a sample rate"""
        self.SP = SoundParam
        self.onset = None
        self.signal = signal
        self.sr = sr
        self.range = freq_range
        self.trimmed = None
        self.noise = None

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

    # noinspection PyUnresolvedReferences
    def plot(self, kind, **kwargs):
        """
        General plotting method for the Signal class, supported plots are:

        'signal' :
            Plots the time varying real signal as amplitude vs time.

        'envelop' :
            Plots the envelop of the signal as amplitude vs time.

        'log envelop' :
            Plots the envelop with logarithmic window widths on a logarithmic x axis scale.

        'fft' :
            Plots the Fourier Transform of the Signal.
            If `ticks = 'bins'` is supplied in the keyword arguments, the frequency ticks are replaced
            with the frequency bin values

        'fft hist' :
            Plots the octave based Fourier Transform Histogram.
            Both axes are on a log scale.
            If `ticks = 'bins'` is supplied in the keyword arguments, the frequency ticks are replaced
            with the frequency bin values

        'peaks' :
            Plots the Fourier Transform of the Signal, with the peaks detected with the `Signal.peaks()` method.
            If `peak_height = True` is supplied in the keyword arguments the computed height threshold is
            shown on the plot.

        'peak damping' :
            Plots the frequency vs damping scatter of the damping ratio computed from the
            Fourier Transform peak shapes. A polynomial curve fit is added to help visualisation.
            Supported key word arguments are :
            `n=5` : The order of the fitted polynomial curve, default is 5,
            if the supplied value is too high, it will be reduced until the number of peaks
            is sufficient to fit the polynomial.

            `inverse=True` : Default value is True, if False, the damping ratio is shown instead
            of its inverse.

            `normalize=False` : Default value is False, if True the damping values are normalized
            from 0 to 1, to help analyze results and compare Sounds.

            `ticks=None` : Default value is None, if `ticks='bins'` the x axis ticks are replaced with
            frequency bin values.

        'time damping' :
            Shows the signal envelop with the fitted negative exponential curve used to determine the
            time damping ratio of the signal.
        """
        illegal_kwargs = ['max_time', 'n', 'ticks', 'normalize', 'inverse', 'peak_height']
        plot_kwargs = {i: kwargs[i] for i in kwargs if i not in illegal_kwargs}

        if kind == 'signal':

            plt.plot(self.time(), self.signal, alpha=0.6, **plot_kwargs)
            plt.xlabel('time (s)')
            plt.ylabel('amplitude')
            plt.grid('on')

        elif kind == 'envelop':

            plt.plot(self.envelop_time(), self.envelop(), **plot_kwargs)
            plt.xlabel("time (s)")
            plt.ylabel("amplitude")
            plt.grid('on')

        elif kind == 'log envelop':
            log_envelop, log_envelop_time = self.log_envelop()
            if ('max_time' in kwargs.keys()) and (kwargs['max_time'] < log_envelop_time[-1]):
                max_index = np.nonzero(log_envelop_time >= kwargs['max_time'])[0][0]
            else:
                max_index = len(log_envelop_time)

            plt.plot(log_envelop_time[:max_index], log_envelop[:max_index], **plot_kwargs)
            plt.xlabel("time (s)")
            plt.ylabel("amplitude")
            plt.xscale('log')
            plt.grid('on')

        elif kind == 'fft':
            # find the index corresponding to the fft range
            result = np.where(self.fft_frequencies() >= self.SP.general.fft_range.value)[0]
            if len(result) == 0:
                last_index = -1
            else:
                last_index = result[0]
            plt.plot(self.fft_frequencies()[:last_index], self.fft()[:last_index], **plot_kwargs)
            plt.xlabel("frequency"),
            plt.ylabel("amplitude"),
            plt.yscale('log')
            plt.grid('on')

            if 'ticks' in kwargs and kwargs['ticks'] == 'bins':
                labels = [label for label in self.SP.bins.__dict__ if label != 'name']
                labels.append('brillance')
                x = [param.value for param in self.SP.bins.__dict__.values() if param != 'bins']
                x.append(11250)
                x_formatter = matplotlib.ticker.FixedFormatter(labels)
                x_locator = matplotlib.ticker.FixedLocator(x)
                ax = plt.gca()
                ax.xaxis.set_major_locator(x_locator)
                ax.xaxis.set_major_formatter(x_formatter)
                ax.tick_params(axis="x", labelrotation=90)

        elif kind == 'fft hist':
            # Histogram of frequency values occurences in octave bins
            plt.hist(self.fft_bins(), utils.octave_histogram(self.SP.general.octave_fraction.value),
                     alpha=0.7, **plot_kwargs)
            plt.xlabel('Fréquence (Hz)')
            plt.ylabel('Amplitude')
            plt.xscale('log')
            plt.yscale('log')
            plt.grid('on')

            if 'ticks' in kwargs and kwargs['ticks'] == 'bins':
                labels = [label for label in self.SP.bins.__dict__ if label != 'name']
                labels.append('brillance')
                x = [param.value for param in self.SP.bins.__dict__.values() if param != 'bins']
                x.append(11250)
                x_formatter = matplotlib.ticker.FixedFormatter(labels)
                x_locator = matplotlib.ticker.FixedLocator(x)
                ax = plt.gca()
                ax.xaxis.set_major_locator(x_locator)
                ax.xaxis.set_major_formatter(x_formatter)
                ax.tick_params(axis="x", labelrotation=90)

        elif kind == 'peaks':
            fft_freqs = self.fft_frequencies()
            fft = self.fft()
            max_index = np.where(fft_freqs >= self.SP.general.fft_range.value)[0][0]
            peak_indexes, height = self.peaks(height=True)
            plt.xlabel('Fréquence (Hz)')
            plt.ylabel('Amplitude')
            plt.yscale('log')
            plt.grid('on')

            if 'color' not in plot_kwargs.keys():
                plot_kwargs['color'] = 'k'
            plt.plot(fft_freqs[:max_index], fft[:max_index], **plot_kwargs)
            plt.scatter(fft_freqs[peak_indexes], fft[peak_indexes], color='r')
            if ('peak_height' in kwargs.keys()) and (kwargs['peak_height']):
                plt.plot(fft_freqs[:max_index], height, color='r')

        elif kind == 'peak damping':
            # Get the damping ration and peak frequencies
            if 'inverse' in kwargs.keys() and kwargs['inverse'] is False:
                zetas = np.array(self.peak_damping())
                ylabel = r'Damping $\zeta$'
            else:
                zetas = 1 / np.array(self.peak_damping())
                ylabel = r'Inverse Damping $1/\zeta$'

            peak_freqs = self.fft_frequencies()[self.peaks()]

            # If a polynomial order is supplied assign it, if not default is 5
            if 'n' in kwargs.keys():
                n = kwargs['n']
            else:
                n = 5

            # If labels are supplied the default color are used
            if 'label' in plot_kwargs:
                plot_kwargs['color'] = None
                plot2_kwargs = plot_kwargs.copy()
                plot2_kwargs['label'] = None

            # If not black and red are used
            else:
                plot_kwargs['color'] = 'r'
                plot2_kwargs = plot_kwargs.copy()
                plot2_kwargs['color'] = 'k'

            if 'normalize' in kwargs.keys() and kwargs['normalize']:
                zetas = np.array(zetas)/np.array(zetas).max()

            plt.scatter(peak_freqs, zetas, **plot_kwargs)
            fun = utils.nth_order_polynomial_fit(n, peak_freqs, zetas)
            freq = np.linspace(peak_freqs[0], peak_freqs[-1], 100)
            plt.plot(freq, fun(freq), **plot2_kwargs)
            plt.grid('on')
            plt.title('Frequency vs Damping Factor with Order ' + str(n))
            plt.xlabel('Frequency (Hz)')
            plt.ylabel(ylabel)

            if 'ticks' in kwargs and kwargs['ticks'] == 'bins':
                labels = [label for label in self.SP.bins.__dict__ if label != 'name']
                labels.append('brillance')
                x = [param.value for param in self.SP.bins.__dict__.values() if param != 'bins']
                x.append(11250)
                x_formatter = matplotlib.ticker.FixedFormatter(labels)
                x_locator = matplotlib.ticker.FixedLocator(x)
                ax = plt.gca()
                ax.xaxis.set_major_locator(x_locator)
                ax.xaxis.set_major_formatter(x_formatter)
                ax.tick_params(axis="x", labelrotation=90)

        elif kind == 'time damping':
            # Get the envelop data
            envelop_time = self.normalize().envelop_time()
            envelop = self.normalize().envelop()

            # First point is the maximum because e^-kt is stricly decreasing
            first_index = np.argmax(envelop)

            # The second point is the first point where the signal crosses the lower_threshold line
            second_point_thresh = self.SP.damping.lower_threshold.value

            try:
                second_index = np.flatnonzero(envelop[first_index:] <= second_point_thresh)[0]
            except IndexError:
                second_index = np.flatnonzero(envelop[first_index:] <= second_point_thresh * 2)[0]

            # Function to compute the residual for the exponential curve fit
            def residual_function(zeta_w, t, s):
                return np.exp(zeta_w[0] * t) - s
            zeta_guess = [-0.5]

            result = scipy.optimize.least_squares(residual_function, zeta_guess,
                                                  args=(envelop_time[first_index:second_index],
                                                        envelop[first_index:second_index]))
            # Get the zeta*omega constant
            zeta_omega = result.x[0]

            # Compute the fundamental frequency in radiants of the signal
            wd = 2 * np.pi * self.fundamental()

            # Plot the two points used for the regression
            plt.scatter(envelop_time[[first_index, second_index]], envelop[[first_index, second_index]], color='r')

            # get the current ax
            ax = plt.gca()

            # Plot the damping curve
            ax.plot(envelop_time[first_index:second_index],
                    np.exp(zeta_omega*envelop_time[first_index:second_index]), c='k')

            plt.sca(ax)
            self.normalize().plot('envelop', **plot_kwargs)

            if 'label' not in plot_kwargs.keys():
                ax.legend(['damping curve', 'signal envelop'])

            title = 'Zeta : ' + str(np.around(-zeta_omega/wd, 5)) + ' Fundamental ' + \
                    str(np.around(self.fundamental(), 0)) + 'Hz'
            plt.title(title)

    def fft(self):
        """
        Computes the Fast Fourier Transform of the signal and returns the vector.
        :return: Fast Fourier Transform amplitude values in a numpy array
        """
        fft = np.fft.fft(self.signal)
        fft = np.abs(fft[:int(len(fft) // 2)])  # Only the symmetric of the absolute value
        return fft / np.max(fft)

    def peaks(self, max_freq=None, height=False, result=False):
        """
        Computes the harmonic peaks indexes from the FFT of the signal
        :param max_freq: Supply a max frequency value overiding the one in guitarsounds_parameters
        :param height: if True the height threshold is returned to be used in the 'peaks' plot
        :param result: if True the Scipy peak finding results dictionary is returned
        :return: peak indexes
        """
        # Replace None by the default value
        if max_freq is None:
            max_freq = self.SP.general.fft_range.value

        # Get the fft and fft frequencies from the signal
        fft, fft_freq = self.fft(), self.fft_frequencies()

        # Find the max index
        max_index = np.where(fft_freq >= max_freq)[0][0]

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
        if number_of_peaks > 0:
            average_len = int(max_index / number_of_peaks) * 3
        else:
            average_len = int(max_index / 3)

        if average_len % 2 == 0:
            average_len += 1

        average_fft = sig.savgol_filter(fft[:max_index], average_len, 1, mode='mirror') * 1.9
        min_freq_index = np.where(fft_freq >= 70)[0][0]
        average_fft[:min_freq_index] = 1

        peak_indexes, res = sig.find_peaks(fft[:max_index], height=average_fft, distance=min_freq_index)

        # Remove noisy peaks at the low frequencies
        while fft[peak_indexes[0]] < 5e-2:
            peak_indexes = np.delete(peak_indexes, 0)
        while fft[peak_indexes[-1]] < 1e-4:
            peak_indexes = np.delete(peak_indexes, -1)

        if not height and not result:
            return peak_indexes
        elif height:
            return peak_indexes, average_fft
        elif result:
            return peak_indexes, res
        elif height and result:
            return peak_indexes, height, res

    def time_damping(self):
        """
        Computes the time wise damping ratio of the signal by fitting a negative exponential curve
        to the Signal envelop and computing the ratio with the Signal fundamental frequency.
        :return: The damping ratio, a scalar.
        """
        # Get the envelop data
        envelop_time = self.normalize().envelop_time()
        envelop = self.normalize().envelop()

        # First point is the maximum because e^-kt is stricly decreasing
        first_index = np.argmax(envelop)

        # The second point is the first point where the signal crosses the lower_threshold line
        second_point_thresh = self.SP.damping.lower_threshold.value
        try:
            second_index = np.flatnonzero(envelop[first_index:] <= second_point_thresh)[0]
        except IndexError:
            second_index = np.flatnonzero(envelop[first_index:] <= second_point_thresh*2)[0]

        # Function to compute the residual for the exponential curve fit
        def residual_function(zeta_w, t, s):
            """
            Function computing the residual to curve fit a negative exponential to the signal envelop
            :param zeta_w: zeta*omega constant
            :param t: time vector
            :param s: signal
            :return: residual
            """
            return np.exp(zeta_w[0] * t) - s

        zeta_guess = [-0.5]

        result = scipy.optimize.least_squares(residual_function, zeta_guess,
                                              args=(envelop_time[first_index:second_index],
                                                    envelop[first_index:second_index]))
        # Get the zeta*omega constant
        zeta_omega = result.x[0]

        # Compute the fundamental frequency in radiants of the signal
        wd = 2 * np.pi * self.fundamental()
        return -zeta_omega / wd

    def peak_damping(self):
        """
        Computes the frequency wise damping with the half bandwidth method on the Fourier Transform peaks
        :return: an array containing the peak damping values
        """
        zetas = []
        fft_freqs = self.fft_frequencies()
        fft = self.fft()[:len(fft_freqs)]
        for peak in self.peaks():
            peak_frequency = fft_freqs[peak]
            peak_height = fft[peak]
            root_height = peak_height / np.sqrt(2)
            frequency_roots = scipy.interpolate.InterpolatedUnivariateSpline(fft_freqs, fft - root_height).roots()
            sorted_roots_indexes = np.argsort(np.abs(frequency_roots - peak_frequency))
            w2, w1 = frequency_roots[sorted_roots_indexes[:2]]
            w1, w2 = np.sort([w1, w2])
            zeta = (w2 - w1) / (2 * peak_frequency)
            zetas.append(zeta)
        return np.array(zetas)

    def fundamental(self):
        """
        Returns the fundamental approximated by the first peak of the fft
        :return: fundamental value (Hz)
        """
        index = self.peaks()[0]
        fundamental = self.fft_frequencies()[index]
        return fundamental

    def cavity_peak(self):
        """
        Finds the Hemlotz cavity frequency index from the Fourier Transform by searching for a peak in the expected
        range (80 - 100 Hz), if the fundamental is too close to the expected Hemlotz frequency a comment
        is printed and None is returned.
        :return: If successful the cavity peak index
        """
        first_index = np.where(self.fft_frequencies() >= 80)[0][0]
        second_index = np.where(self.fft_frequencies() >= 110)[0][0]
        cavity_peak = np.argmax(self.fft()[first_index:second_index]) + first_index
        if self.fundamental() == self.fft_frequencies()[cavity_peak]:
            print('Cavity peak is obscured by the fundamental')
        else:
            return cavity_peak

    def cavity_frequency(self):
        """
        Finds the hemlotz cavity frequency from the Fourier Transform by searching for a peak in the expected
        range (80 - 100 Hz), if the fundamental is too close to the expected hemlotz frequency a comment
        is printed and None is returned.
        :return: If successful, the cavity peak frequency
        """
        first_index = np.where(self.fft_frequencies() >= 80)[0][0]
        second_index = np.where(self.fft_frequencies() >= 110)[0][0]
        cavity_peak = np.argmax(self.fft()[first_index:second_index]) + first_index
        if self.fundamental() == self.fft_frequencies()[cavity_peak]:
            print('Cavity peak is obscured by the fundamental')
            return 0
        else:
            return self.fft_frequencies()[cavity_peak]

    def fft_frequencies(self):
        """
        Computes the frequency vector associated to the Signal Fourier Transform
        :return: an array containing the frequency values.
        """
        fft = self.fft()
        fft_frequencies = np.fft.fftfreq(len(fft) * 2, 1 / self.sr)  # Frequencies corresponding to the bins
        return fft_frequencies[:len(fft)]

    def fft_bins(self):
        """
        Transforms the Fourier Transform signal into a statistic distribution.
        Occurences of each frequency varies from 0 to 100 according to their
        amplitude.
        :return : a list containing the frequency occurences.
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
        Method calculating the amplitude envelope of a signal as a
        maximum of the absolute value of the signal.
        :return: Amplitude envelop of the signal
        """
        # Get the hop length
        hop_length = self.SP.envelop.hop_length.value

        # Compute the envelop
        envelop = np.array(
            [np.max(np.abs(self.signal[i:i + self.SP.envelop.frame_size.value])) for i in
             range(0, len(self.signal), hop_length)])

        envelop = np.insert(envelop, 0, 0)
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
        """
        Computes the logarithmic scale envelop of the signal.
        The width of the samples increases exponentially so that
        the envelop appears having a constant window width on
        an X axis logarithmic scale.
        :return: The log envelop and the time vector associated in a tuple
        """
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

        return np.array(log_envelop), np.array(log_envelop_time)

    def find_onset(self, verbose=True):
        """
        Finds the onset as an increase in more of 50% with the maximum normalized value above 0.5
        :param verbose: Prints a warning if the algorithm does not converge
        :return: the index of the onset in the signal
        """
        # Index corresponding to the onset time interval
        window_index = np.ceil(self.SP.onset.onset_time.value * self.sr).astype(int)
        # Use the normalized signal to compare against a fixed value
        onset_signal = self.normalize()
        overlap = window_index // 2  # overlap for algorithm progression
        # Initial values
        increase = 0
        i = 0
        broke = False
        while increase <= 0.5:
            signal_min = np.min(np.abs(onset_signal.signal[i:i + window_index]))
            signal_max = np.max(np.abs(onset_signal.signal[i:i + window_index]))
            if (signal_max > 0.5) and (signal_min != 0):
                increase = signal_max / signal_min
            else:
                increase = 0
            i += overlap
            if i + window_index > len(self.signal):
                if verbose:
                    print('Onset detection did not converge \n')
                    print('Approximating onset with signal max value \n')
                    broke = True
                    break
        if broke:
            return np.argmax(self.signal)
        else:
            return np.argmax(np.abs(self.signal[i:i + window_index])) + i

    def trim_onset(self, verbose=True):
        """
        Trim the signal at the onset (max) minus the delay in milliseconds as
        Specified in the SoundParameters
        :param : verbose if False the warning comments are not displayed
        :return : a trimmed signal with a noise attribute
        """
        # nb of samples to keep before the onset
        delay_samples = int((self.SP.onset.onset_delay.value / 1000) * self.sr)
        onset = self.find_onset(verbose=verbose)  # find the onset

        if onset > delay_samples:  # To make sure the index is positive
            trimmed_signal = Signal(self.signal[onset - delay_samples:], self.sr, self.SP)
            trimmed_signal.noise = self.signal[:onset - delay_samples]
            trimmed_signal.trimmed = True
            trimmed_signal.onset = np.argmax(trimmed_signal.signal)
            return trimmed_signal

        else:
            if verbose:
                print('Signal is too short to be trimmed before onset.')
                print('')
            self.trimmed = False
            return self

    def trim_time(self, time_length):
        """
        Trims the signal to the specified length and returns a new Signal instance.
        :param time_length: desired length of the new signal in seconds.
        :return: A trimmed Signal
        """
        max_index = int(time_length * self.sr)
        time_trimmed_signal = Signal(self.signal[:max_index], self.sr, self.SP)
        time_trimmed_signal.time_length = time_length
        return time_trimmed_signal

    def filter_noise(self, verbose=True):
        """
        Method filtering the noise from the recorded signal and returning a filtered signal.
        If the signal was not trimmed it is trimmed in place then filtered.
        If the signal can not be trimmed it can't be filtered and the original signal is returned
        :return : A Signal instance, filtered if possible.
        """
        try:
            return Signal(reduce_noise(audio_clip=self.signal, noise_clip=self.noise), self.sr, self.SP)
        except AttributeError:
            if self.trimmed is False:
                if verbose:
                    print('Not sufficient noise in the raw signal, unable to filter.')
                    print('')
                return self

    def normalize(self):
        """
        Normalizes the signal to [-1, 1] and returns the normalised instance.
        :return : A normalized signal
        """
        factor = np.max(np.abs(self.signal))
        normalised_signal = Signal((self.signal / factor), self.sr, self.SP)
        normalised_signal.norm_factor = (1 / factor)
        return normalised_signal

    def make_freq_bins(self):
        """
        Method to divide a signal in frequency bins using butterworth filters
        bins are passed as a dict, default values are :
        - bass < 100 Hz
        - mid = 100 - 700 Hz
        - highmid = 700 - 2000 Hz
        - uppermid = 2000 - 4000 Hz
        - presence = 4000 - 6000 Hz
        - brillance > 6000 Hz
        :return : A dictionary with the divided signal as values and bin names as keys
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

    def save_wav(self, name, path=''):
        """
        Create a soundfile from a signal
        :param name: the name of the saved file
        :param path: the path were the '.wav' file is saved
        """
        write(path + name + ".wav", self.signal, self.sr)


class Sound(object):
    """
    A class to store audio signals obtained from a sound and compare them
    """

    def __init__(self, file, name='', fundamental=None, SoundParams=None):
        """
        Creates a Sound instance from a .wav file, name as a string and fundamental frequency
        value can be user specified.
        :param file: file path to the .wav file
        :param name: Sound instance name to use in plot legend and titles
        :param fundamental: Fundamental frequency value if None the value is estimated
        from the FFT (see `Signal.fundamental`).
        :param SoundParams: SoundParameters to use in the Sound instance
        """
        # create a reference of the parameters
        if SoundParams is None:
            self.SP = SP
        else:
            self.SP = SoundParams

        if type(file) == str:
            # Load the soundfile using librosa
            signal, sr = librosa.load(file)
            self.file = file

        elif type(file) == tuple:
            signal, sr = file

        # create a Signal class from the signal and sample rate
        self.raw_signal = Signal(signal, sr, self.SP)

        # Allow user specified fundamental
        self.fundamental = fundamental
        self.name = name

    def condition(self, verbose=True, return_self=False):
        """
        A method conditioning the Sound instance.
        - Trimming to just before the onset
        - Filtering the noise
        :param verbose: if True problem with trimming and filtering are reported
        :param return_self: If True the method returns the conditioned Sound instance
        :return: a conditioned Sound instance if `return_self = True`
        """
        self.trim_signal(verbose=verbose)
        self.filter_noise(verbose=verbose)
        self.bin_divide()
        if self.fundamental is None:
            self.fundamental = self.signal.fundamental()
        if return_self:
            return self

    def use_raw_signal(self, normalized=False):
        """
        Assigns the raw signal to the `signal` attribute of the Sound instance to
        analyze it
        :param normalized: if True, the raw signal is first normalized
        :return: None
        """
        if normalized:
            self.signal = self.raw_signal.normalize()
        else:
            self.signal = self.raw_signal

    def bin_divide(self):
        """
        Calls the `.make_freq_bins` method of the signal to create the signals associated
        to the frequency bins. The bins are all stored in the `.bin` attribute and also as
        their names (Ex: `Sound.mid` contains the mid signal).
        :return: None
        """
        """ a method to divide the main signal into frequency bins"""
        # divide in frequency bins
        self.bins = self.signal.make_freq_bins()
        # unpack the bins
        self.bass, self.mid, self.highmid, self.uppermid, self.presence, self.brillance = self.bins.values()

    def filter_noise(self, verbose=True):
        """
        Filters the noise in the signal attribute
        :param verbose: if True problem are printed to the terminal
        :return: None
        """
        # filter the noise in the Signal class
        self.signal = self.trimmed_signal.filter_noise(verbose=verbose)

    def trim_signal(self, verbose=True):
        """
        A method to trim the signal to a specific time before the onset. The time value
        can be changed in the SoundParameters.
        :param verbose: if True problems encountered are printed to the terminal
        :return: None
        """
        # Trim the signal in the signal class
        self.trimmed_signal = self.raw_signal.trim_onset(verbose=verbose)

    def validate_trim(self):
        """
        Graphic validation of the `.trim_onset` method.
        Used to see if the signal onset was determined accurately
        :return: None
        """
        if hasattr(self, 'trimmed_signal'):
            fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 6))
            ax1.plot(self.raw_signal.envelop_time(), self.raw_signal.envelop(), color='k')
            ax1.set(title='Old Envelop', xlabel='time', ylabel='amplitude')
            ax2.plot(self.trimmed_signal.envelop_time(), self.trimmed_signal.envelop(), color='k')
            onset_index = self.trimmed_signal.onset
            ax2.scatter(self.trimmed_signal.time()[onset_index], self.trimmed_signal.signal[onset_index], color='r')
            ax2.set(title='Trimmed signal', xlabel='time', ylabel='amplitude')
            plt.tight_layout()
        else:
            print('signal was not trimmed')

    def validate_noise(self):
        """
        Audio validation of the `.filter_noise()` method.
        Allows the user to listen to the filtered and unfiltered signals
        in a jupyter notebook
        :return: None
        """
        if hasattr(self, 'trimmed_signal'):
            print('not filtered')
            self.trimmed_signal.listen()
            print('filtered')
            self.signal.listen()
        else:
            print('signal was not filtered')

    def listen_freq_bins(self):
        """
        Method to listen to all the frequency bins of a sound
        in a Jupyter Notebook
        :return : None
        """
        for key in self.bins.keys():
            print(key)
            self.bins[key].listen()

    def plot_freq_bins(self, bins=None):
        """
        Method to plot all the frequency bins of a sound
        :param bins: frequency bins as a list to plot on the graph
        if none are specified, all the bins are plotted.
        """
        if bins is None:
            bins = self.bins.keys()
        if bins[0] == 'all':
            bins = self.bins.keys()

        for key in bins:
            lab = key + ' : ' + str(int(self.bins[key].range[0])) + ' - ' + str(int(self.bins[key].range[1])) + ' Hz'
            self.bins[key].plot('log envelop', label=lab)
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(fontsize="x-small")  # using a named size

    def peak_damping(self):
        peak_indexes = self.signal.peaks()
        frequencies = self.signal.fft_frequencies()[peak_indexes]
        damping = self.signal.peak_damping()
        table_data = np.array([frequencies, np.array(damping) * 100]).transpose()
        print(tabulate(table_data, headers=['Frequency (Hz)', 'Damping ratio (%)']))
