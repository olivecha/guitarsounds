from soundfile import write
import IPython.display as ipd
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np
import os
import scipy
import scipy.optimize
import scipy.integrate
import scipy.interpolate
from scipy.integrate import trapezoid
from scipy import signal as sig
from guitarsounds.parameters import sound_parameters
import guitarsounds.utils as utils
from tabulate import tabulate
import wave


"""
Classes
"""

class SoundPack(object):
    """
    A class to store and analyse multiple sounds
    Some methods are only available for SoundPacks containing two sounds
    """

    def __init__(self, *sounds, names=None, fundamentals=None, 
                 SoundParams=None, equalize_time=True):
        """
        The SoundPack can be instantiated from existing Sound class instances, 
        either in a list or as multiple arguments
        The class can also handle the creation of Sound class instances if the 
        arguments are filenames, either a list or multiple arguments.

        :param sounds: `guitarsounds.Sound` instaces or filenames either as 
        multiple arguments or as a list
        :param names: list of strings with the names of the sounds that will be 
        used in the plot legend labels 
        :param fundamentals: list of numbers corresponding to the known sound 
        fundamental frequencies. 
        :param SoundParams: `guitarsounds.SoundParams` instance used to get 
        the parameters used in the computation of the sound attributes 
        :param equalize_time: if True, all the sounds used to create the 
        SoundPack are truncated to the length of the shortest sound. 

        If the number of Sound contained is equal to two, the SoundPack will 
        be 'dual' and the associated methods will be available :

            - `SoundPack.compare_peaks`
            - `SoundPack.fft_mirror`
            - `SoundPack.fft_diff`
            - `SoundPack.integral_compare`

        If it contains multiple sounds the SoundPack will be multiple and a 
        reduced number of methods will be available to call

        If the fundamental frequency is supplied for each sound, the 
        computation of certain features can be more efficient, such as the 
        time damping computation or Hemlotz frequency computation.

        Examples :
        ```python
        Sound_Test = SoundPack('sounds/test1.wav', 'sounds/test2.wav', names=['A', 'B'], fundamentals = [134, 134])

        sounds = [sound1, sound2, sound3, sound4, sound5] # instances of the Sound class
        large_Test = SoundPack(sounds, names=['1', '2', '3', '4', '5'])
        ```
        """
        # create a copy of the sound parameters
        if SoundParams is None:
            self.SP = sound_parameters()
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

        # If filenames are supplied
        if type(sounds[0]) is str:
            self.sounds_from_files(sounds, names=names, fundamentals=fundamentals)

        # Else sound instances are supplied
        else:
            self.sounds = sounds

            # sound name defined in constructor
            if names and (len(names) == len(self.sounds)):
                for sound, n in zip(self.sounds, names):
                    sound.name = n

            else:
                # names obtained from the supplied sounds
                names = [sound.name for sound in self.sounds if sound.name]

                # all sounds have a name
                if len(names) == len(sounds):
                    self.names = names

                # Assign a default value to names
                else:
                    names = [str(n) for n in np.arange(1, len(sounds) + 1)]
                    for sound, n in zip(self.sounds, names):
                        sound.name = n

            # If the sounds are not conditionned condition them
            for s in self.sounds:
                if ~hasattr(s, 'signal'):
                    s.condition()

        if equalize_time:
            self.equalize_time()

        # Define bin strings
        self.bin_strings = [*list(self.SP.bins.__dict__.keys())[1:], 'brillance']

        # Sort according to fundamental
        key = np.argsort([sound.fundamental for sound in self.sounds])
        self.sounds = np.array(self.sounds)[key]

    def sounds_from_files(self, sound_files, names=None, fundamentals=None):
        """
        Create Sound class instances and assign them to the SoundPack 
        from a list of files

        :param sound_files: sound filenames
        :param names: sound names
        :param fundamentals: user specified fundamental frequencies
        :return: None
        """
        # Make the default name list from sound filenames if none is supplied
        if (names is None) or (len(names) != len(sound_files)):
            names = [os.path.split(file)[-1][:-4] for file in sound_files]  # remove the .wav

        # If the fundamentals are not supplied or mismatch in number None is used
        if (fundamentals is None) or (len(fundamentals) != len(sound_files)):
            fundamentals = len(sound_files) * [None]

        # Create Sound instances from files
        self.sounds = []
        for file, name, fundamental in zip(sound_files, names, fundamentals):
            self.sounds.append(Sound(file, name=name, fundamental=fundamental,
                                     SoundParams=self.SP))

    def equalize_time(self):
        """
        Trim the sounds so that they all have the length of the shortest sound, 
        trimming is done at the end of the sounds.
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
        Normalize all the signals in the SoundPack and returns a normalized
        instance of itself. See `Signal.normalize` for more information.
        :return: SoundPack with normalized signals
        """
        new_sounds = []
        names = [sound.name for sound in self.sounds]
        fundamentals = [sound.fundamental for sound in self.sounds]
        for sound in self.sounds:
            sound.signal = sound.signal.normalize()
            new_sounds.append(sound)

        self.sounds = new_sounds

        return self

    """
    Methods for all SoundPacks
    """

    def plot(self, kind, **kwargs):
        """
        Superimposed plot of all the sounds on one figure for a specific kind

        :param kind: feature name passed to the `signal.plot()` method
        :param kwargs: keywords arguments to pass to the `matplotlib.plot()` 
        method
        :return: None

        __ Multiple SoundPack Method __
        Plots a specific signal.plot for all sounds on the same figure

        Ex : SoundPack.plot('fft') plots the fft of all sounds on a single figure
        The color argument is set to none so that the plots have different colors
        """
        plt.figure(figsize=(8, 6))
        for sound in self.sounds:
            kwargs['label'] = sound.name
            kwargs['color'] = None
            sound.signal.plot.method_dict[kind](**kwargs)
        ax = plt.gca()
        ax.set_title(kind + ' plot')
        ax.legend()
        return ax

    def compare_plot(self, kind, **kwargs):
        """
        Plots all the sounds on different figures to compare them for a specific kind

        :param kind: feature name passed to the `signal.plot()` method
        :param kwargs: keywords arguments to pass to the `matplotlib.plot()` 
        method
        :return: None

        __ Multiple SoundPack Method __
        Draws the same kind of plot on a different axis for each sound
        Example : `SoundPack.compare_plot('peaks')` with 4 Sounds will plot a 
        figure with 4 axes, with each a different 'peak' plot.
        """
        # if a dual SoundPack : only plot two big plots
        if self.kind == 'dual':
            fig, axs = plt.subplots(1, 2, figsize=(12, 4))
            for sound, ax in zip(self.sounds, axs):
                plt.sca(ax)
                sound.signal.plot.method_dict[kind](**kwargs)
                ax.set_title(kind + ' ' + sound.name)
            plt.tight_layout()

        # If a multiple SoundPack : plot on a grid of axes
        elif self.kind == 'multiple':
            # find the n, m values for the subplots line and columns
            n = len(self.sounds)
            cols = 0
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

            fig, axs = plt.subplots(rows, cols, figsize=(12, 4 * rows))
            axs = axs.reshape(-1)
            for sound, ax in zip(self.sounds, axs):
                plt.sca(ax)
                sound.signal.plot.method_dict[kind](**kwargs)
                title = ax.get_title()
                title = sound.name + ' ' + title
                ax.set_title(title)

            if remainder != 0:
                for ax in axs[-(cols - remainder):]:
                    ax.set_axis_off()
            plt.tight_layout()
        else:
            raise Exception
        return axs

    def freq_bin_plot(self, f_bin='all'):
        """
        Plots the log envelope of specified frequency bins
        :param f_bin: frequency bins to compare, Supported arguments are :
        'all', 'bass', 'mid', 'highmid', 'uppermid', 'presence', 'brillance'

        __ Multiple SoundPack Method __
        A function to compare signals decomposed frequency wise in the time 
        domain on a logarithm scale. The methods plot all the sounds and plots 
        their frequency bins according to the frequency bin argument f_bin.

        Example : SoundPack.freq_bin_plot(f_bin='mid') will plot the log-scale 
        envelope of the 'mid' signal of every sound in the SoundPack.
        """

        if f_bin == 'all':
            # Create one plot per bin
            fig, axs = plt.subplots(3, 2, figsize=(12, 12))
            axs = axs.reshape(-1)
            for key, ax in zip([*list(self.SP.bins.__dict__.keys())[1:], 'brillance'], axs):
                plt.sca(ax)
                # plot every sound for a frequency bin
                norm_factors = np.array([son.bins[key].normalize().norm_factor for son in self.sounds])
                for i, son in enumerate(self.sounds):
                    son.bins[key].normalize().old_plot('log envelope', label=son.name)
                plt.xscale('log')
                plt.legend()
                son = self.sounds[-1]
                title0 = ' ' + key + ' : ' + str(int(son.bins[key].range[0])) + ' - ' + str(
                         int(son.bins[key].range[1])) + ' Hz, '
                title1 = 'Norm. Factors : '
                title2 = 'x, '.join(str(np.around(norm_factor, 0)) for norm_factor in norm_factors)
                plt.title(title0 + title1 + title2)
            plt.tight_layout()

        elif f_bin in [*list(sound_parameters().bins.__dict__.keys())[1:], 'brillance']:
            plt.figure(figsize=(10, 4))
            # Plot every envelope for a single frequency bin
            norm_factors = np.array([son.bins[f_bin].normalize().norm_factor for son in self.sounds])
            for i, son in enumerate(self.sounds):
                son.bins[f_bin].normalize().old_plot('log envelope', label=(str(i + 1) + '. ' + son.name))
            plt.xscale('log')
            plt.legend()
            son = self.sounds[-1]
            title0 = ' ' + f_bin + ' : ' + str(int(son.bins[f_bin].range[0])) + ' - ' + str(
                int(son.bins[f_bin].range[1])) + ' Hz, '
            title1 = 'Norm. Factors : '
            title2 = 'x, '.join(str(np.around(norm_factor, 0)) for norm_factor in norm_factors)
            plt.title(title0 + title1 + title2)

        else:
            print('invalid frequency bin')

    def fundamentals(self):
        """
        Displays the fundamentals of every sound in the SoundPack

        :return: None

        __ Multiple Soundpack Method __
        """
        names = np.array([sound.name for sound in self.sounds])
        fundamentals = np.array([np.around(sound.fundamental, 1) for sound in self.sounds])
        key = np.argsort(fundamentals)
        table_data = [names[key], fundamentals[key]]

        table_data = np.array(table_data).transpose()

        print(tabulate(table_data, headers=['Name', 'Fundamental (Hz)']))

    def integral_plot(self, f_bin='all'):
        """
        Normalized cumulative bin power plot for the frequency bins.
        See `Plot.integral` for more information.

        :param f_bin: frequency bins to compare, Supported arguments are 
        'all', 'bass', 'mid', 'highmid', 'uppermid', 'presence', 'brillance'

        __ Multiple SoundPack Method __
        Plots the cumulative integral plot of specified frequency bins
        see help(Plot.integral)
        """

        if f_bin == 'all':
            # create a figure with 6 axes
            fig, axs = plt.subplots(3, 2, figsize=(12, 12))
            axs = axs.reshape(-1)

            for key, ax in zip(self.bin_strings, axs):
                plt.sca(ax)
                norm_factors = np.array([son.bins[key].normalize().norm_factor for son in self.sounds])
                for sound in self.sounds:
                    sound.bins[key].plot.integral(label=sound.name)
                plt.legend()
                sound = self.sounds[-1]
                title0 = ' ' + key + ' : ' + str(int(sound.bins[key].range[0])) + ' - ' + str(
                    int(sound.bins[key].range[1])) + ' Hz, '
                title1 = 'Norm. Factors : '
                title2 = 'x, '.join(str(np.around(norm_factor, 0)) for norm_factor in norm_factors)
                plt.title(title0 + title1 + title2)
                plt.title(title0 + title1 + title2)
            plt.tight_layout()

        elif f_bin in self.bin_strings:
            fig, ax = plt.subplots(figsize=(6, 4))
            plt.sca(ax)
            norm_factors = np.array([son.bins[f_bin].normalize().norm_factor for son in self.sounds])
            for sound in self.sounds:
                sound.bins[f_bin].plot.integral(label=sound.name)
            plt.legend()
            sound = self.sounds[-1]
            title0 = ' ' + f_bin + ' : ' + str(int(sound.bins[f_bin].range[0])) + ' - ' + str(
                int(sound.bins[f_bin].range[1])) + ' Hz, '
            title1 = 'Norm. Factors : '
            title2 = 'x, '.join(str(np.around(norm_factor, 0)) for norm_factor in norm_factors)
            plt.title(title0 + title1 + title2)

        else:
            print('invalid frequency bin')
            
    def bin_power_table(self):
        """
        Displays a table with the signal power contained in every frequency bin
        
        The power is computed as the time integral of the signal, such as :

            $ P = \int_0^{t_{max}} sig(t) dt $
        
        __ Multiple SoundPack Method __
        The sounds are always normalized before computing the power. Because 
        the signal amplitude is normalized between -1 and 1, the power value 
        is adimentional, and can only be used to compare two sounds between  
        eachother.
        """
        # Bin power distribution table
        bin_strings = self.bin_strings
        integrals = []

        # for every sound in the SoundPack
        for sound in self.sounds:

            integral = []
            # for every frequency bin in the sound
            for f_bin in bin_strings:
                log_envelope, log_time = sound.bins[f_bin].normalize().log_envelope()
                integral.append(scipy.integrate.trapezoid(log_envelope, log_time))

            # a list of dict for every sound
            integrals.append(integral)

        # make the table
        table_data = np.array([list(bin_strings), *integrals]).transpose()
        sound_names = [sound.name for sound in self.sounds]

        print('___ Signal Power Frequency Bin Distribution ___ \n')
        print(tabulate(table_data, headers=['bin', *sound_names]))

    def bin_power_hist(self):
        """
        Histogram of the frequency bin power for multiple sounds

        The power is computed as the time integral of the signal, such as :

            $ P = \int_0^{t_{max}} sig(t) dt $
        
        __ Multiple SoundPack Method __
        The sounds are always normalized before computing the power. Because 
        the signal amplitude is normalized between -1 and 1, the power value 
        is adimentional, and can only be used to compare two sounds between  
        eachother.
        """
        # Compute the bin powers
        bin_strings = self.bin_strings
        integrals = []

        # for every sound in the SoundPack
        for sound in self.sounds:

            integral = []
            # for every frequency bin in the sound
            for f_bin in bin_strings:
                log_envelope, log_time = sound.bins[f_bin].normalize().log_envelope()
                integral.append(trapezoid(log_envelope, log_time))

            # a list of dict for every sound
            integral = np.array(integral)
            integral /= np.max(integral)
            integrals.append(integral)

        # create the bar plotting vectors
        fig, ax = plt.subplots(figsize=(6, 6))

        # make the bar plot
        n = len(self.sounds)
        width = 0.8 / n
        # get nice colors
        cmap = matplotlib.cm.get_cmap('Set2')
        for i, sound in enumerate(self.sounds):
            x = np.arange(i * width, len(bin_strings) + i * width)
            y = integrals[i]
            if n < 8:
                color = cmap(i)
            else:
                color = None

            if i == n // 2:
                ax.bar(x, y, width=width, tick_label=list(bin_strings), label=sound.name, color=color)
            else:
                ax.bar(x, y, width=width, label=sound.name, color=color)
        ax.set_xlabel('frequency bin name')
        ax.set_ylabel('normalized power')
        plt.legend()
    
    def listen(self):
        """
        Listen to all the sounds in the SoundPack inside the Jupyter Notebook 
        environment

        __ Multiple SoundPack Method __
        """
        for sound in self.sounds:
            sound.signal.listen()
        
    """
    Methods for dual SoundPacks
    """

    def compare_peaks(self):
        """
        Plot to compare the FFT peaks values of two sounds

        __ Dual SoundPack Method __
        Compares the peaks in the Fourier Transform of two Sounds,
        the peaks having the highest difference are highlighted. If no peaks
        are found to have a significant difference, only the two normalized 
        Fourier transform are plotted in a mirror configuration to visualize
        them.
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

            short_peaks_1 = [peak for peak in peaks1 if peak < (len(freq1) - 1)]
            short_peaks_2 = [peak for peak in peaks2 if peak < (len(freq1) - 1)]
            peak_distance1 = np.mean([freq1[peaks1[i]] - freq1[peaks1[i + 1]] for i in range(len(short_peaks_1) - 1)]) / 4
            peak_distance2 = np.mean([freq2[peaks2[i]] - freq2[peaks2[i + 1]] for i in range(len(short_peaks_2) - 1)]) / 4
            peak_distance = np.abs(np.mean([peak_distance1, peak_distance2]))

            # Align  the two peak vectors
            new_peaks1 = []
            new_peaks2 = []
            for peak1 in short_peaks_1:
                for peak2 in short_peaks_2:
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
                if np.isclose(difference_threshold, 0.):
                    break

            # Plot the output
            plt.figure(figsize=(10, 6))
            plt.yscale('symlog', linthresh=10e-1)

            # Sound 1
            plt.plot(freq1, fft1, color='#919191', label=son1.name)
            plt.scatter(freq1[new_peaks1], fft1[new_peaks1], color='b', label='peaks')
            if len(different_peaks1) > 0:
                plt.scatter(freq1[different_peaks1[0]], fft1[different_peaks1[0]], color='g', label='diff peaks')
                annotation_string = 'Peaks with ' + str(np.around(difference_threshold, 2)) + ' difference'
                plt.annotate(annotation_string, (freq1[different_peaks1[0]] + peak_distance / 2, fft1[different_peaks1[0]]))

            # Sound2
            plt.plot(freq2, -fft2, color='#3d3d3d', label=son2.name)
            plt.scatter(freq2[new_peaks2], -fft2[new_peaks2], color='b')
            if len(different_peaks2) > 0:
                plt.scatter(freq2[different_peaks2[0]], -fft2[different_peaks2[0]], color='g')
            plt.title('Fourier Transform Peak Analysis for ' + son1.name + ' and ' + son2.name)
            plt.grid('on')
            plt.legend()
            ax = plt.gca()
            ax.set_xlabel('frequency (Hz)')
            ax.set_ylabel('mirror amplitude (0-1)')
        else:
            raise ValueError('Unsupported for multiple sounds SoundPacks')

    def fft_mirror(self):
        """
        Plot the Fourier Transforms of two sounds on opposed axis to compare the spectral content

        __ Dual SoundPack Method __
        The fourier transforms are plotted normalized between 0 and 1. 
        The y scale is symmetric logarithmic, so that a signal is plotted 
        between 0 and -1, and the other is plotted between 0 and 1. 
        :return: None
        """
        if self.kind == 'dual':
            son1 = self.sounds[0]
            son2 = self.sounds[1]
            fft_range_value = sound_parameters().general.fft_range.value
            fft_freq_value = son1.signal.fft_frequencies()
            index = np.where(fft_freq_value > fft_range_value)[0][0]

            plt.figure(figsize=(10, 6))
            plt.yscale('symlog')
            plt.grid('on')
            plt.plot(son1.signal.fft_frequencies()[:index], son1.signal.fft()[:index], label=son1.name)
            plt.plot(son2.signal.fft_frequencies()[:index], -son2.signal.fft()[:index], label=son2.name)
            plt.xlabel('frequency (Hz)')
            plt.ylabel('mirror amplitude (normalized)')
            plt.title('Mirror Fourier Transform for ' + son1.name + ' and ' + son2.name)
            plt.legend()

        else:
            print('Unsupported for multiple sounds SoundPacks')

    def fft_diff(self, fraction=3, ticks=None):
        """
        Plot the difference between the spectral distribution in the two sounds

        :param fraction: octave fraction value used to compute the frequency bins A higher number will show
        a more precise comparison, but conclusions may be harder to draw.
        :param ticks:  If equal to 'bins' the frequency bins intervals are used as X axis ticks
        :return: None

        __ Dual SoundPack Method __
        Compare the Fourier Transform of two sounds by computing the differences of the octave bins heights.
        The two FTs are superimposed on the first plot to show differences
        The difference between the two FTs is plotted on the second plot
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
            ax1.set_xlabel('frequency (Hz)')
            ax1.set_ylabel('amplitude')
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
            ax2.set_xlabel('frequency (Hz)')
            ax2.set_ylabel('<- sound 2 : sound 1 ->')
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

    def integral_compare(self, f_bin='all'):
        """
          Cumulative bin envelope integral comparison for two signals

          :param f_bin: frequency bins to compare, Supported arguments are :
          'all', 'bass', 'mid', 'highmid', 'uppermid', 'presence', 'brillance'
          :return: None

          __ Dual SoundPack Method __
          Plots the cumulative integral plot of specified frequency bins
          and their difference as surfaces
          """

        # Case when plotting all the frequency bins
        if f_bin == 'all':
            fig, axs = plt.subplots(3, 2, figsize=(16, 16))
            axs = axs.reshape(-1)

            # get the bins frequency values
            self.bin_strings = self.sounds[0].bins.keys()
            bins1 = self.sounds[0].bins.values()
            bins2 = self.sounds[1].bins.values()

            for signal1, signal2, bin_string, ax in zip(bins1, bins2, self.bin_strings, axs):
                # Compute the log time and envelopes integrals
                log_envelope1, log_time1 = signal1.normalize().log_envelope()
                log_envelope2, log_time2 = signal2.normalize().log_envelope()
                env_range1 = np.arange(2, len(log_envelope1), 1)
                env_range2 = np.arange(2, len(log_envelope2), 1)
                integral1 = np.array([trapezoid(log_envelope1[:i], log_time1[:i]) for i in env_range1])
                integral2 = np.array([trapezoid(log_envelope2[:i], log_time2[:i]) for i in env_range2])
                time1 = log_time1[2:len(log_time1):1]
                time2 = log_time2[2:len(log_time2):1]

                # resize arrays to match shape
                common_len = min(len(time1), len(time2))
                time1 = time1[:common_len]
                time2 = time2[:common_len]
                integral1 = integral1[:common_len]
                integral2 = integral2[:common_len]
                # Normalize
                max_value = np.max(np.hstack([integral1, integral2]))
                integral1 /= max_value
                integral2 /= max_value

                # plot the integral area curves
                ax.fill_between(time1, integral1, label=self.sounds[0].name, alpha=0.4)
                ax.fill_between(time2, -integral2, label=self.sounds[1].name, alpha=0.4)
                ax.fill_between(time2, integral1 - integral2, color='g', label='int diff', alpha=0.6)
                ax.set_xlabel('time (s)')
                ax.set_ylabel('mirror cumulative power (normalized)')
                ax.set_xscale('log')
                ax.set_title(bin_string)
                ax.legend()
                ax.grid('on')

            plt.tight_layout()

        elif f_bin in self.bin_strings:

            # Compute the log envelopes and areau curves
            signal1 = self.sounds[0].bins[f_bin]
            signal2 = self.sounds[1].bins[f_bin]
            log_envelope1, log_time1 = signal1.normalize().log_envelope()
            log_envelope2, log_time2 = signal2.normalize().log_envelope()
            integral1 = np.array([trapezoid(log_envelope1[:i], log_time1[:i]) for i in np.arange(2, len(log_envelope1), 1)])
            integral2 = np.array([trapezoid(log_envelope2[:i], log_time2[:i]) for i in np.arange(2, len(log_envelope2), 1)])
            time1 = log_time1[2:len(log_time1):1]
            time2 = log_time2[2:len(log_time2):1]

            # resize arrays to match shape
            common_len = min(len(time1), len(time2))
            time1 = time1[:common_len]
            time2 = time2[:common_len]
            integral1 = integral1[:common_len]
            integral2 = integral2[:common_len]
            # Normalize
            max_value = np.max(np.hstack([integral1, integral2]))
            integral1 /= max_value
            integral2 /= max_value

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.fill_between(time1, integral1, label=self.sounds[0].name, alpha=0.4)
            ax.fill_between(time2, -integral2, label=self.sounds[1].name, alpha=0.4)
            ax.fill_between(time2, integral1 - integral2, color='g', label='int diff', alpha=0.6)

            ax.set_xlabel('time (s)')
            ax.set_ylabel('mirror cumulative power (normalized)')
            ax.set_xscale('log')
            ax.set_title(f_bin)
            ax.legend(loc='upper left')
            ax.grid('on')

        else:
            print('invalid frequency bin')


class Sound(object):
    """
    A class to store audio signals obtained from a sound and compare them
    """

    def __init__(self, data, name='', condition=True, auto_trim=False,
                 use_raw_signal=False, normalize_raw_signal=False, 
                 fundamental=None, SoundParams=None):
        """
        Creates a Sound instance from a .wav file, name as a string and 
        fundamental frequency value can be user specified.

        :param data: data path to the .wav data
        :param name: Sound instance name to use in plot legend and titles
        :param condition: Bool, whether to condition or not the Sound instance 
        if `True`, the `Sound` instance is conditioned in the constructor
        :param auto_trim: Bool, whether to trim the end of the sound or not 
        according to predefined sound length correlated to the fundamental.
        :param use_raw_signal: Do not condition the `Sound` and instead 
        use the raw signal
        :param normalize_raw_signal: If `use_raw_signal` is `True`, setting 
        `normalize_raw_signal` to `True` will normalize the raw signal before it
        is used in the `Sound` class
        :param fundamental: Fundamental frequency value if None the value is 
        estimated
        from the FFT (see `Signal.fundamental`).
        :param SoundParams: SoundParameters to use in the Sound instance
        """
        # create a reference of the parameters
        if SoundParams is None:
            self.SP = sound_parameters()
        else:
            self.SP = SoundParams

        if type(data) == str:
            # Load the sound data using librosa
            if data.split('.')[-1] != 'wav':
                raise ValueError('Only .wav are supported')
            else:
                signal, sr = utils.load_wav(data)

        elif type(data) == tuple:
            signal, sr = data

        else:
            raise TypeError

        # create a Signal class from the signal and sample rate
        self.raw_signal = Signal(signal, sr, self.SP)
        # create an empty plot attribute
        self.plot = None
        # Allow user specified fundamental
        self.fundamental = fundamental
        self.name = name
        # create an empty signal attribute
        self.signal = None
        self.trimmed_signal = None
        self.bins = None
        self.bass = None
        self.mid = None
        self.highmid = None
        self.uppermid = None
        self.presence = None
        self.brillance = None

        if condition:
            self.condition(verbose=True,
                           return_self=False,
                           auto_trim=auto_trim,
                           resample=True)
        else:
            if use_raw_signal:
                self.use_raw_signal(normalized=normalize_raw_signal,
                                    return_self=False)

    def condition(self, verbose=True, return_self=False, auto_trim=False, resample=True):
        """
        A method conditioning the Sound instance.
        - Trimming to just before the onset
        :param verbose: if True problem with trimming are reported
        :param return_self: If True the method returns the conditioned Sound instance
        :param auto_trim: If True, the sound is trimmed to a fixed length according to its fundamental
        :param resample: If True, the signal is resampled to 22050 Hz
        :return: a conditioned Sound instance if `return_self = True`
        """
        # Resample only if the sample rate is not 22050
        if resample & (self.raw_signal.sr != 22050):
            signal, sr = self.raw_signal.signal, self.raw_signal.sr
            self.raw_signal = Signal(utils.resample(signal, sr, 22050), 22050, self.SP)

        self.trim_signal(verbose=verbose)
        self.signal = self.trimmed_signal
        if self.fundamental is None:
            self.fundamental = self.signal.fundamental()
        if auto_trim:
            time = utils.freq2trim(self.fundamental)
            self.signal = self.signal.trim_time(time)
        self.plot = self.signal.plot
        self.bin_divide()
        if return_self:
            return self

    def use_raw_signal(self, normalized=False, return_self=False):
        """
        Assigns the raw signal to the `signal` attribute of the Sound instance to
        analyze it
        :param normalized: if True, the raw signal is first normalized
        :param return_self: if True the Sound instance is return after the signal attribute is defined
        :return: None, self if return_self is True
        """
        if normalized:
            self.signal = self.raw_signal.normalize()
        else:
            self.signal = self.raw_signal
        self.bin_divide()
        if return_self:
            return self

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

    def trim_signal(self, verbose=True):
        """
        A method to trim the signal to a specific time before the onset. The time value
        can be changed in the SoundParameters.
        :param verbose: if True problems encountered are printed to the terminal
        :return: None
        """
        # Trim the signal in the signal class
        self.trimmed_signal = self.raw_signal.trim_onset(verbose=verbose)

    def listen_freq_bins(self):
        """
        Method to listen to all the frequency bins of a sound

        The bins signals are obtained by filtering the sound signal
        with band pass filters.

        See guitarsounds.parameters.sound_parameters().bins.info() for the
        frequency bin intervals.
        """
        for key in self.bins.keys():
            print(key)
            self.bins[key].normalize().listen()

    def plot_freq_bins(self, bins='all'):
        """
        Method to plot all the frequency bins logarithmic envelopes of a sound

        The parameter `bins` allows choosing specific frequency bins to plot
        By default the function plots all the bins
        Supported bins arguments are :
        'all', 'bass', 'mid', 'highmid', 'uppermid', 'presence', 'brillance'

        Example :
        `Sound.plot_freq_bins(bins='all')` plots all the frequency bins
        `Sound.plot_freq_bins(bins=['bass', 'mid'])` plots the bass and mid bins
        """

        if bins[0] == 'all':
            bins = 'all'

        if bins == 'all':
            bins = self.bins.keys()

        for key in bins:
            lab = key + ' : ' + str(int(self.bins[key].range[0])) + ' - ' + str(int(self.bins[key].range[1])) + ' Hz'
            self.bins[key].old_plot('log envelope', label=lab)

        plt.xscale('log')
        plt.yscale('log')
        plt.legend(fontsize="x-small")  # using a named size

    def peak_damping(self):
        """
        Prints a table with peak damping values and peak frequency values

        The peaks are found with the `signal.peaks()` function and the damping
        values are computed with the half power bandwidth method.
        """
        peak_indexes = self.signal.peaks()
        frequencies = self.signal.fft_frequencies()[peak_indexes]
        damping = self.signal.peak_damping()
        table_data = np.array([frequencies, np.array(damping) * 100]).transpose()
        print(tabulate(table_data, headers=['Frequency (Hz)', 'Damping ratio (%)']))

    def bin_hist(self):
        """
        Histogram of the frequency bin power

        frequency bin power is computed as the integral of the bin envelope.
        See guitarsounds.parameters.sound_parameters().bins.info() for the
        frequency bin intervals.
        """
        # Compute the bin powers
        bin_strings = list(self.bins.keys())
        integrals = []

        for f_bin in bin_strings:
            log_envelope, log_time = self.bins[f_bin].normalize().log_envelope()
            integral = trapezoid(log_envelope, log_time)
            integrals.append(integral)
        max_value = np.max(integrals)
        integrals = np.array(integrals)/max_value

        # create the bar plotting vectors
        fig, ax = plt.subplots(figsize=(6, 6))

        x = np.arange(0, len(bin_strings))
        y = integrals
        ax.bar(x, y, tick_label=list(bin_strings))
        ax.set_xlabel("frequency bin name")
        ax.set_ylabel("frequency bin power (normalized)")


class Signal(object):
    """
    A Class to do computations on an audio signal.

    The signal is never changed in the class, when transformations are made, a new instance is returned.
    """

    def __init__(self, signal, sr, SoundParam, freq_range=None):
        """ 
        Create a Signal class from a vector of samples and a sample rate
        :param signal: vector containing the signal samples
        :param sr: sample rate of the signal (Hz)
        :param SoundParam: Sound Parameter instance to use with the signal
        :para freq_range: Frequency range to use with the signal.
        This is the maximum frequency used when computing Fourier transform
        peaks and plotting Fourier transform related plots.
        """
        self.SP = SoundParam
        self.onset = None
        self.signal = signal
        self.sr = sr
        self.range = freq_range
        self.plot = Plot()
        self.plot.parent = self
        self.norm_factor = None

    def time(self):
        """
        Returns the time vector associated with the signal
        :return: numpy array corresponding to the time values
        of the signal samples in seconds
        """
        return np.linspace(0,
                           len(self.signal) * (1 / self.sr),
                           len(self.signal))

    def listen(self):
        """
        Method to listen the sound signal in a Jupyter Notebook

        Listening to the sounds imported in the analysis tool allows the
        user to validate if the sound was well trimmed and filtered

        A temporary file is created, the IPython display Audio function is
        called on it and then the file is removed
        """
        file = 'temp.wav'
        write(file, self.signal, self.sr)
        ipd.display(ipd.Audio(file))
        os.remove(file)

    def old_plot(self, kind, **kwargs):
        """
        Convenience function for the different signal plots

        Calls the function corresponding to Plot.kind()
        See help(guitarsounds.analysis.Plot) for info on the different plots
        """
        self.plot.method_dict[kind](**kwargs)

    def fft(self):
        """
        Computes the Fast Fourier Transform of the signal and returns the vector.
        :return: Fast Fourier Transform amplitude values in a numpy array
        """
        fft = np.fft.fft(self.signal)
        fft = np.abs(fft[:int(len(fft) // 2)])  # Only the symmetric part of the absolute value
        return fft / np.max(fft)

    def spectral_centroid(self):
        """
        Spectral centroid of the frequency content of the signal
        :return: Spectral centroid of the signal (float)
        """
        SC = np.sum(self.fft() * self.fft_frequencies()) / np.sum(self.fft())
        return SC


    def peaks(self, max_freq=None, height=False, result=False):
        """
        Computes the harmonic peaks indexes from the FFT of the signal
        :param max_freq: Supply a max frequency value overriding the one in
        guitarsounds_parameters
        :param height: if True the height threshold is returned to be used
        in the 'peaks' plot
        :param result: if True the Scipy peak finding results dictionary
        is returned
        :return: peak indexes
        """
        # Replace None by the default value
        if max_freq is None:
            max_freq = self.SP.general.fft_range.value

        # Get the fft and fft frequencies from the signal
        fft, fft_freq = self.fft(), self.fft_frequencies()

        # Find the max index
        try:
            max_index = np.where(fft_freq >= max_freq)[0][0]
        except IndexError:
            max_index = fft_freq.shape[0] 

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
        to the Signal envelope and computing the ratio with the Signal fundamental frequency.
        :return: The damping ratio, a scalar.
        """
        # Get the envelope data
        envelope, envelope_time = self.normalize().envelope() 

        # First point is the maximum because e^-kt is strictly decreasing
        first_index = np.argmax(envelope)

        # The second point is the first point where the signal crosses the lower_threshold line
        second_point_thresh = self.SP.damping.lower_threshold.value
        try:
            second_index = np.flatnonzero(envelope[first_index:] <= second_point_thresh)[0]
        except IndexError:
            second_index = np.flatnonzero(envelope[first_index:] <= second_point_thresh * 2)[0]

        # Function to compute the residual for the exponential curve fit
        def residual_function(zeta_w, t, s):
            """
            Function computing the residual to curve fit a negative exponential to the signal envelope
            :param zeta_w: zeta*omega constant
            :param t: time vector
            :param s: signal
            :return: residual
            """
            return np.exp(zeta_w[0] * t) - s

        zeta_guess = [-0.5]

        result = scipy.optimize.least_squares(residual_function, zeta_guess,
                                              args=(envelope_time[first_index:second_index],
                                                    envelope[first_index:second_index]))
        # Get the zeta*omega constant
        zeta_omega = result.x[0]

        # Compute the fundamental frequency in radians of the signal
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
        :return: The index of the cavity peak
        """
        first_index = np.where(self.fft_frequencies() >= 80)[0][0]
        second_index = np.where(self.fft_frequencies() >= 110)[0][0]
        cavity_peak = np.argmax(self.fft()[first_index:second_index]) + first_index
        return cavity_peak

    def cavity_frequency(self):
        """
        Finds the Hemlotz cavity frequency from the Fourier Transform by searching for a peak in the expected
        range (80 - 100 Hz), if the fundamental is too close to the expected Hemlotz frequency a comment
        is printed and None is returned.
        :return: If successful, the cavity peak frequency
        """
        cavity_peak = self.cavity_peak()
        if self.fundamental() == self.fft_frequencies()[cavity_peak]:
            print('Cavity peak is obscured by the fundamental')
            return 0
        else:
            return self.fft_frequencies()[cavity_peak]

    def fft_frequencies(self):
        """
        Computes the frequency vector associated with the Signal Fourier Transform
        :return: an array containing the frequency values.
        """
        fft = self.fft()
        fft_frequencies = np.fft.fftfreq(len(fft) * 2, 1 / self.sr)  # Frequencies corresponding to the bins
        return fft_frequencies[:len(fft)]

    def fft_bins(self):
        """
        Transforms the Fourier transform array into a statistic distribution 
        ranging from 0 to 100. 
        Accordingly, the maximum of the Fourier transform with value 1.0 will
        be equal to 100 and casted as an integer.
        Values below 0.001 will be equal to 0. 
        This representation of the Fourier transform is used to construct
        octave bands histograms.
        :return : a list containing the frequency occurrences.
        """

        # Make the FT values integers
        fft_integers = [int(np.around(sample * 100, 0)) for sample in self.fft()]

        # Create a list of the frequency occurrences in the signal
        occurrences = []
        for freq, count in zip(self.fft_frequencies(), fft_integers):
            occurrences.append([freq] * count)

        # flatten the list
        return [item for sublist in occurrences for item in sublist]

    def envelope(self, window=None, overlap=None):
        """
        Method calculating the amplitude envelope of a signal as a
        maximum of the absolute value of the signal. 
        The same `window` and `overlap` parameters should be used to compute 
        the signal and time arrays so that they contain the same 
        number of points (and can be plotted together).
        :param window: integer, length in samples of the window used to
        compute the signal envelope.
        :param overlap: integer, overlap in samples used to overlap two
        subsequent windows in the computation of the signal envelope.
        The overlap value should be smaller than the window value.
        :return: Amplitude envelope of the signal
        """
        if window is None:
            window = self.SP.envelope.frame_size.value
        if overlap is None:
            overlap = window // 2
        elif overlap >= window:
            raise ValueError('Overlap must be smaller than window')
        signal_array = np.abs(self.signal)
        t = self.time()
        # Empty envelope and envelope time
        env = [0]
        env_time = [0]
        idx = 0
        while idx + window < signal_array.shape[0]:
            env.append(np.max(signal_array[idx:idx + window]))
            pt_idx = np.argmax(signal_array[idx:idx + window]) + idx
            env_time.append(t[pt_idx])
            idx += overlap
        _, unique_time_index = np.unique(env_time, return_index=True)
        return np.array(env)[unique_time_index], np.unique(env_time)

    def log_envelope(self):
        """
        Computes the logarithmic scale envelope of the signal.
        The width of the samples increases exponentially so that
        the envelope appears having a constant window width on
        an X axis logarithmic scale.
        :return: The log envelope and the time vector associated in a tuple
        """
        if self.onset is None:
            onset = np.argmax(np.abs(self.signal))
        else:
            onset = self.onset

        start_time = self.SP.log_envelope.start_time.value
        while start_time > (onset / self.sr):
            start_time /= 10.

        start_exponent = int(np.log10(start_time))  # closest 10^x value for smooth graph

        if self.SP.log_envelope.min_window.value is None:
            min_window = 15 ** (start_exponent + 4)
            if min_window < 15:  # Value should at least be 10
                min_window = 15
        else:
            min_window = self.SP.log_envelope.min_window.value

        # initial values
        current_exponent = start_exponent
        current_time = 10 ** current_exponent  # start time on log scale
        index = int(current_time * self.sr)  # Start at the specified time
        window = min_window  # number of samples per window
        overlap = window // 2
        log_envelope = [0]
        log_envelope_time = [0]  # First value for comparison

        while index + window <= len(self.signal):

            while log_envelope_time[-1] < 10 ** (current_exponent + 1):
                if (index + window) < len(self.signal):
                    log_envelope.append(np.max(np.abs(self.signal[index:index + window])))
                    pt_idx = np.argmax(np.abs(self.signal[index:index + window]))
                    log_envelope_time.append(self.time()[index + pt_idx])
                    index += overlap
                else:
                    break

            if window * 10 < self.SP.log_envelope.max_window.value:
                window = window * 10
            else:
                window = self.SP.log_envelope.max_window.value
            overlap = window // 2
            current_exponent += 1
        time, idxs = np.unique(log_envelope_time, return_index=True)
        return np.array(log_envelope)[idxs], time

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
            return np.argmax(np.abs(self.signal))
        else:
            i -= overlap
            return np.argmax(np.abs(self.signal[i:i + window_index])) + i 

    def trim_onset(self, verbose=True):
        """
        Trim the signal at the onset (max) minus the delay in milliseconds as
        Specified in the SoundParameters
        :param verbose: if False the warning comments are not displayed
        :return : The trimmed signal
        """
        # nb of samples to keep before the onset
        delay_samples = int((self.SP.onset.onset_delay.value / 1000) * self.sr)
        onset = self.find_onset(verbose=verbose)  # find the onset

        if onset > delay_samples:  # To make sure the index is positive
            new_signal = self.signal[onset - delay_samples:]
            new_signal[:delay_samples // 2] = new_signal[:delay_samples // 2] * np.linspace(0, 1, delay_samples // 2) 
            trimmed_signal = Signal(new_signal, self.sr, self.SP)
            trimmed_signal.onset = trimmed_signal.find_onset(verbose=verbose)
            return trimmed_signal

        else:
            if verbose:
                print('Signal is too short to be trimmed before onset.')
                print('')
            return self

    def trim_time(self, time_length):
        """
        Trims the signal to the specified length and returns a new Signal instance.
        :param time_length: desired length of the new signal in seconds.
        :return: A trimmed Signal
        """
        max_index = int(time_length * self.sr)
        new_signal = self.signal[:max_index]
        new_signal[-50:] = new_signal[-50:] * np.linspace(1, 0, 50)
        time_trimmed_signal = Signal(new_signal, self.sr, self.SP)
        return time_trimmed_signal

    def normalize(self):
        """
        Normalizes the signal to [-1, 1] and returns the normalized instance.
        return : A normalized signal
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


class Plot(object):
    """
        A class to handle all the plotting functions of the Signal and to allow a nice call signature :
        Signal.plot.envelope()

        Supported plots are :
        'signal', 'envelope', 'log envelope', 'fft', 'fft hist', 'peaks',
        'peak damping', 'time damping', 'integral'
    """

    # Illegal plot key word arguments
    illegal_kwargs = ['max_time', 'n', 'ticks', 'normalize', 'inverse',
                      'peak_height', 'fill']

    def __init__(self):
        # define the parent attribute
        self.parent = None

        # dictionary with methods and keywords
        self.method_dict = {'signal': self.signal,
                            'envelope': self.envelope,
                            'log envelope': self.log_envelope,
                            'fft': self.fft,
                            'fft hist': self.fft_hist,
                            'peaks': self.peaks,
                            'peak damping': self.peak_damping,
                            'time damping': self.time_damping,
                            'integral': self.integral, }

    def sanitize_kwargs(self, kwargs):
        """
        Remove illegal keywords to supply the key word arguments to matplotlib
        :param kwargs: a dictionary of key word arguments
        :return: sanitized kwargs
        """
        return {i: kwargs[i] for i in kwargs if i not in self.illegal_kwargs}

    def set_bin_ticks(self):
        """ Applies the frequency bin ticks to the current plot """
        labels = [label for label in self.parent.SP.bins.__dict__ if label != 'name']
        labels.append('brillance')
        x = [param.value for param in self.parent.SP.bins.__dict__.values() if param != 'bins']
        x.append(11025)
        x_formatter = ticker.FixedFormatter(labels)
        x_locator = ticker.FixedLocator(x)
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_locator)
        ax.xaxis.set_major_formatter(x_formatter)
        ax.tick_params(axis="x", labelrotation=90)

    def signal(self, **kwargs):
        """ Plots the time varying real signal as amplitude vs time. """
        plot_kwargs = self.sanitize_kwargs(kwargs)
        plt.plot(self.parent.time(), self.parent.signal, alpha=0.6, **plot_kwargs)
        plt.xlabel('time (s)')
        plt.ylabel('amplitude [-1, 1]')
        plt.grid('on')

    def envelope(self, **kwargs):
        """
        Plots the envelope of the signal as amplitude vs time.
        """
        plot_kwargs = self.sanitize_kwargs(kwargs)
        envelope_arr, envelope_time = self.parent.envelope()
        plt.plot(envelope_time, envelope_arr, **plot_kwargs)
        plt.xlabel("time (s)")
        plt.ylabel("amplitude [0, 1]")
        plt.grid('on')

    def log_envelope(self, **kwargs):
        """
        Plots the signal envelope with logarithmic window widths on a logarithmic x-axis scale.
        :param max_time: maximum time used for the x-axis in the plot (seconds)
        """
        plot_kwargs = self.sanitize_kwargs(kwargs)
        log_envelope, log_envelope_time = self.parent.log_envelope()

        if ('max_time' in kwargs.keys()) and (kwargs['max_time'] < log_envelope_time[-1]):
            max_index = np.nonzero(log_envelope_time >= kwargs['max_time'])[0][0]
        else:
            max_index = len(log_envelope_time)

        plt.plot(log_envelope_time[:max_index], log_envelope[:max_index], **plot_kwargs)
        plt.xlabel("time (s)")
        plt.ylabel("amplitude [0, 1]")
        plt.xscale('log')
        plt.grid('on')

    def fft(self, **kwargs):
        """
        Plots the Fourier Transform of the Signal.

        If `ticks = 'bins'` is supplied in the keyword arguments, the frequency ticks are replaced
        with the frequency bin values.
        """

        plot_kwargs = self.sanitize_kwargs(kwargs)

        # find the index corresponding to the fft range
        fft_frequencies = self.parent.fft_frequencies()
        fft_range = self.parent.SP.general.fft_range.value
        result = np.where(fft_frequencies >= fft_range)[0]
        if len(result) == 0:
            last_index = len(fft_frequencies)
        else:
            last_index = result[0]

        plt.plot(self.parent.fft_frequencies()[:last_index],
                 self.parent.fft()[:last_index],
                 **plot_kwargs)
        plt.xlabel("frequency (Hz)"),
        plt.ylabel("amplitude (normalized)"),
        plt.yscale('log')
        plt.grid('on')

        if 'ticks' in kwargs and kwargs['ticks'] == 'bins':
            self.set_bin_ticks()

    def fft_hist(self, **kwargs):
        """
            Plots the octave based Fourier Transform Histogram.
            Both axes are on a log scale.

            If `ticks = 'bins'` is supplied in the keyword arguments, the frequency ticks are replaced
            with the frequency bin values
            """

        plot_kwargs = self.sanitize_kwargs(kwargs)

        # Histogram of frequency values occurrences in octave bins
        plt.hist(self.parent.fft_bins(), utils.octave_histogram(self.parent.SP.general.octave_fraction.value),
                 alpha=0.7, **plot_kwargs)
        plt.xlabel('frequency (Hz)')
        plt.ylabel('amplitude (normalized)')
        plt.xscale('log')
        plt.yscale('log')
        plt.grid('on')

        if 'ticks' in kwargs and kwargs['ticks'] == 'bins':
            self.set_bin_ticks()

    def peaks(self, **kwargs):
        """
            Plots the Fourier Transform of the Signal, with the peaks detected
            with the `Signal.peaks()` method.

            If `peak_height = True` is supplied in the keyword arguments the
            computed height threshold is
            shown on the plot.
            """

        plot_kwargs = self.sanitize_kwargs(kwargs)

        fft_freqs = self.parent.fft_frequencies()
        fft = self.parent.fft()
        fft_range = self.parent.SP.general.fft_range.value
        max_index = np.where(fft_freqs >= fft_range)[0][0]
        peak_indexes, height = self.parent.peaks(height=True)
        plt.xlabel('frequency (Hz)')
        plt.ylabel('amplitude')
        plt.yscale('log')
        plt.grid('on')

        if 'color' not in plot_kwargs.keys():
            plot_kwargs['color'] = 'k'
        plt.plot(fft_freqs[:max_index], fft[:max_index], **plot_kwargs)
        plt.scatter(fft_freqs[peak_indexes], fft[peak_indexes], color='r')
        if ('peak_height' in kwargs.keys()) and (kwargs['peak_height']):
            plt.plot(fft_freqs[:max_index], height, color='r')

    def peak_damping(self, **kwargs):
        """
            Plots the frequency vs damping scatter of the damping ratio computed from the
            Fourier Transform peak shapes. A polynomial curve fit is added to help visualisation.

            Supported key word arguments are :

            `n=5` : The order of the fitted polynomial curve, default is 5,
            if the supplied value is too high, it will be reduced until the
            number of peaks is sufficient to fit the polynomial.

            `inverse=True` : Default value is True, if False, the damping ratio is shown instead
            of its inverse.

            `normalize=False` : Default value is False, if True the damping values are normalized
            from 0 to 1, to help analyze results and compare Sounds.

            `ticks=None` : Default value is None, if `ticks='bins'` the x-axis ticks are replaced with
            frequency bin values.
            """
        plot_kwargs = self.sanitize_kwargs(kwargs)
        # Get the damping ratio and peak frequencies
        if 'inverse' in kwargs.keys() and kwargs['inverse'] is False:
            zetas = np.array(self.parent.peak_damping())
            ylabel = r'damping $\zeta$'
        else:
            zetas = 1 / np.array(self.parent.peak_damping())
            ylabel = r'inverse damping $1/\zeta$'

        peak_freqs = self.parent.fft_frequencies()[self.parent.peaks()]

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
            zetas = np.array(zetas) / np.array(zetas).max(initial=0)

        plt.scatter(peak_freqs, zetas, **plot_kwargs)
        fun = utils.nth_order_polynomial_fit(n, peak_freqs, zetas)
        freq = np.linspace(peak_freqs[0], peak_freqs[-1], 100)
        plt.plot(freq, fun(freq), **plot2_kwargs)
        plt.grid('on')
        plt.title('Frequency vs Damping Factor with Order ' + str(n))
        plt.xlabel('frequency (Hz)')
        plt.ylabel(ylabel)

        if 'ticks' in kwargs and kwargs['ticks'] == 'bins':
            self.set_bin_ticks()

    def time_damping(self, **kwargs):
        """
        Shows the signal envelope with the fitted negative exponential
        curve used to determine the time damping ratio of the signal.
        """
        plot_kwargs = self.sanitize_kwargs(kwargs)
        # Get the envelope data
        envelope, envelope_time = self.parent.normalize().envelope() 

        # First point is the maximum because e^-kt is strictly decreasing
        first_index = np.argmax(envelope)

        # The second point is the first point where the signal crosses the lower_threshold line
        second_point_thresh = self.parent.SP.damping.lower_threshold.value
        while True:
            try:
                second_index = np.flatnonzero(envelope[first_index:] <= second_point_thresh)[0]
                break
            except IndexError:
                second_point_thresh *= 2
                if second_point_thresh > 1:
                    raise ValueError("invalid second point threshold encountered, something went wrong")

        # Function to compute the residual for the exponential curve fit
        def residual_function(zeta_w, t, s):
            return np.exp(zeta_w[0] * t) - s

        zeta_guess = [-0.5]

        result = scipy.optimize.least_squares(residual_function, zeta_guess,
                                              args=(envelope_time[first_index:second_index],
                                                    envelope[first_index:second_index]))
        # Get the zeta*omega constant
        zeta_omega = result.x[0]

        # Compute the fundamental frequency in radians of the signal
        wd = 2 * np.pi * self.parent.fundamental()

        # Plot the two points used for the regression
        plt.scatter(envelope_time[[first_index, first_index + second_index]], 
                    envelope[[first_index, first_index + second_index]], color='r')

        # get the current ax
        ax = plt.gca()

        # Plot the damping curve
        ax.plot(envelope_time[first_index:second_index + first_index],
                np.exp(zeta_omega * envelope_time[first_index:second_index + first_index]), 
                c='k',
                linestyle='--')

        plt.sca(ax)
        if 'alpha' not in plot_kwargs:
            plot_kwargs['alpha'] = 0.6
        self.parent.normalize().plot.envelope(**plot_kwargs)

        if 'label' not in plot_kwargs.keys():
            ax.legend(['damping curve', 'signal envelope'])

        title = 'Zeta : ' + str(np.around(-zeta_omega / wd, 5)) + ' Fundamental ' + \
                str(np.around(self.parent.fundamental(), 0)) + 'Hz'
        plt.title(title)

    def integral(self, **kwargs):
        """
        Cumulative integral plot of the normalized signal log envelope

        Represents the power distribution variation in time for the signal.
        This is a plot of the function $F(x)$ such as :

        $ F(x) = \int_0^x env(x) dx $

        Where e(x) is the signal envelope.
        """
        # sanitize the kwargs
        plot_kwargs = self.sanitize_kwargs(kwargs)

        # Compute log envelope and log time
        log_envelope, log_time = self.parent.normalize().log_envelope()


        # compute the cumulative integral
        integral = [trapezoid(log_envelope[:i], log_time[:i]) for i in np.arange(2, len(log_envelope), 1)]
        integral /= np.max(integral)

        # plot the integral
        plt.plot(log_time[2:], integral, **plot_kwargs)

        # Add labels and scale
        plt.xlabel('time (s)')
        plt.ylabel('cumulative power (normalized)')
        plt.xscale('log')
        plt.grid('on')


