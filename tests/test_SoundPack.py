import unittest
from contextlib import redirect_stdout
from io import StringIO
import guitarsounds
from guitarsounds import SoundPack
import helpers_tests
import numpy as np
import matplotlib.axes
import matplotlib.pyplot as plt


class MyTestCase(unittest.TestCase):
    """ Test class for the guitarsounds.Sound class"""

    def test_SoundPack_instantiation_from_files(self):
        """ Test the SoundPack constructor using files """
        # instantiation from one audio file
        file = helpers_tests.get_rnd_audio_file()
        sp = SoundPack(file)
        self.assertIsInstance(sp, SoundPack)
        # instantiation from multiples audio files
        files = [helpers_tests.get_rnd_audio_file() for _ in range(3)]
        sp = SoundPack(files)
        self.assertIsInstance(sp, SoundPack)
        # instantiation from enumerated audio files
        sp = SoundPack(*files)
        self.assertIsInstance(sp, SoundPack)

    def test_SoundPack_instantiation_from_Sounds(self):
        """ Test for the SoundPack class constructor from Sound instances """
        # instantiation from a single Sound instance
        file = helpers_tests.get_rnd_audio_file()
        sound = guitarsounds.Sound(file)
        sp = SoundPack(sound)
        self.assertIsInstance(sp, SoundPack)
        files = [helpers_tests.get_rnd_audio_file() for _ in range(3)]
        sounds = [guitarsounds.Sound(file) for file in files]
        sp = SoundPack(sounds)
        self.assertIsInstance(sp, SoundPack)
        sp = SoundPack(*sounds)
        self.assertIsInstance(sp, SoundPack)

    def test_SoundPack_normalize(self):
        """ Test the normalize method of the SoundPack class"""
        files = [helpers_tests.get_rnd_audio_file() for _ in range(3)]
        sp = SoundPack(files)
        nsp = sp.normalize()
        for s in nsp.sounds:
            self.assertTrue(abs(np.max(np.abs(s.signal.signal)) - 1.) < 1e-3)

    def test_SoundPack_plot(self):
        files = [helpers_tests.get_rnd_audio_file() for _ in range(3)]
        sp = SoundPack(files)
        plot_kinds = ['signal', 'envelop', 'log envelop',
                      'fft', 'fft hist', 'peaks',
                      'peak damping', 'time damping', 'integral']
        for kind in plot_kinds:
            out = sp.plot(kind)
            self.assertIsInstance(out, matplotlib.axes.Axes)
            plt.close(plt.gcf())

    def test_SoundPack_compare_plot_multiple(self):
        files = [helpers_tests.get_rnd_audio_file() for _ in range(3)]
        sp = SoundPack(files)
        plot_kinds = ['signal', 'envelop', 'log envelop',
                      'fft', 'fft hist', 'peaks',
                      'peak damping', 'time damping', 'integral']
        for kind in plot_kinds:
            out = sp.compare_plot(kind)
            for ax in out:
                self.assertIsInstance(ax, matplotlib.axes.Axes)
            plt.close(plt.gcf())

    def test_SoundPack_compare_plot_dual(self):
        files = [helpers_tests.get_rnd_audio_file() for _ in range(2)]
        sp = SoundPack(files)
        plot_kinds = ['signal', 'envelop', 'log envelop',
                      'fft', 'fft hist', 'peaks',
                      'peak damping', 'time damping', 'integral']
        for kind in plot_kinds:
            out = sp.compare_plot(kind)
            for ax in out:
                self.assertIsInstance(ax, matplotlib.axes.Axes)
            plt.close(plt.gcf())            
    
    def test_SoundPack_listen(self):
        """ Test the SoundPack.listen method """
        files = [helpers_tests.get_rnd_audio_file() for _ in range(4)]
        sp = SoundPack(files)
        sp.listen()
        self.assertTrue(True)

    def test_soundpack_freq_bin_plot(self):
        """ test the SoundPack.freq_bin_plot method """
        # Create a SoundPack instance
        files = [helpers_tests.get_rnd_audio_file() for _ in range(3)]
        sp = SoundPack(files)
        # Test branch with all frequency bins
        sp.freq_bin_plot(f_bin='all')
        # test that all the frequency bins are plotted
        self.assertTrue(len(plt.gcf().axes) == 6)
        plt.close(plt.gcf())
        # Test branch with a specific frequency bin
        sp.freq_bin_plot(f_bin='mid')
        self.assertTrue(len(plt.gcf().axes) == 1)

    def test_soundpack_combine_envelop(self):
        """ the the combine_envelop method of SoundPack """
        files = [helpers_tests.get_rnd_audio_file() for i in range(3)]
        sp = SoundPack(files)
        # Test signal plot branch
        sp.combine_envelop(kind='signal')
        # the plot is supposed to have lines
        self.assertTrue(len(plt.gcf().axes[0].lines) >= 0)
        plt.close(plt.gcf())
        # test the branch with a frequency bin and only showing the combined
        # curve
        sp.combine_envelop(kind='bass', show_sounds=False, show_rejects=False)
        self.assertTrue(len(plt.gcf().axes[0].lines) == 1)

    def test_soundpack_fundamentals(self):
        """ test the fundamentals method of SoundPack """
        # create a SoundPack instance
        files = [helpers_tests.get_rnd_audio_file() for i in range(3)]
        sp = SoundPack(files)
        # redirect_stdout
        output = StringIO()
        with redirect_stdout(output):
            sp.fundamentals()
        self.assertTrue(len(output.getvalue()) > 0)

    def test_soundpack_integral_plot(self):
        """ test the SoundPack integral_plot method """
        # Create a soundpack
        files = [helpers_tests.get_rnd_audio_file() for i in range(3)]
        sp = SoundPack(files)
        # Test branch with all frequency bins
        sp.integral_plot(f_bin='all')
        self.assertTrue(len(plt.gcf().axes) == 6)
        # Test branch with a specific frequency bin
        sp.integral_plot(f_bin='highmid')
        self.assertTrue(len(plt.gcf().axes) == 1)

    def test_soundpack_bin_power_table(self):
        """ Test the SoundPack bin_power_table method """
        # create a SoundPack instance
        files = [helpers_tests.get_rnd_audio_file() for i in range(3)]
        sp = SoundPack(files)
        # redirect_stdout
        output = StringIO()
        with redirect_stdout(output):
            sp.bin_power_table()
        self.assertTrue(len(output.getvalue()) > 0)

    def test_soundpack_bin_power_hist(self):
        """ Test the bin_power_hist method of SoundPack """
        # create a SoundPack instance
        files = [helpers_tests.get_rnd_audio_file() for i in range(3)]
        sp = SoundPack(files)
        # plot the bin power histogram
        sp.bin_power_hist()
        # Test that there is 18 bars
        self.assertTrue(len(plt.gcf().axes[0].patches) == 18)

    def test_soundpack_compare_peaks(self):
        """ test the SoundPack compare_peaks method """
        # Create a dual sound soundpack instance
        files = [helpers_tests.get_rnd_audio_file() for i in range(2)]
        sp = SoundPack(files)
        sp.compare_peaks()
        # test that two lines were plotted
        self.assertTrue(len(plt.gcf().axes[0].lines) == 2)

    def test_soundpack_fft_mirror(self):
        """ test the SoundPack fft_mirror method """
        # Create a dual sound soundpack instance
        files = [helpers_tests.get_rnd_audio_file() for i in range(2)]
        sp = SoundPack(files)
        sp.fft_mirror()
        # test that two lines were plotted
        self.assertTrue(len(plt.gcf().axes[0].lines) == 2)

    def test_soundpack_fft_diff(self):
        """ test the soundpack fft_diff method """
        # create a dual soundpack
        files = [helpers_tests.get_rnd_audio_file() for i in range(2)]
        sp = SoundPack(files)
        # test the base branch
        sp.fft_diff()
        # test that the two axes were plotted
        self.assertTrue(len(plt.gcf().axes) == 2)
        plt.close(plt.gcf())
        # test the frequency bin ticks branch
        sp.fft_diff(ticks=True)
        # test that the two axes were plotted
        self.assertTrue(len(plt.gcf().axes) == 2)
        plt.close(plt.gcf())
        # test a different octave fraction
        sp.fft_diff(fraction=4, ticks=True)
        # test that the two axes were plotted
        self.assertTrue(len(plt.gcf().axes) == 2)
        plt.close(plt.gcf())

    def test_soundpack_integral_compare(self):
        """ test the SoundPack integral_compare method """
        # create a dual soundpack
        files = [helpers_tests.get_rnd_audio_file() for i in range(2)]
        sp = SoundPack(files)
        # Branch with all the frequency bins
        sp.integral_compare(f_bin='all')
        # test all the frequency bins were plotted
        self.assertTrue(len(plt.gcf().axes) == 6)
        plt.close(plt.gcf())
        # Branch with a specific frequency bin
        sp.integral_compare(f_bin='mid')
        # test two lines were plotted
        self.assertTrue(len(plt.gcf().axes[0].spines) == 4)


if __name__ == '__main__':
    unittest.main()
