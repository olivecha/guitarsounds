import unittest
from guitarsounds import helpers_tests, Sound, utils
import sys
import io
import matplotlib.pyplot as plt
import numpy as np


class MyTestCase(unittest.TestCase):
    """ Test class for the guitarsounds.Sound class"""

    def test_Sound_instantiation(self):
        """ Test for the Sound class constructor"""
        # instantiation from audio file
        file = helpers_tests.get_rnd_audio_file()
        sound = Sound(file)
        self.assertIsInstance(sound, Sound)
        self.assertTrue(hasattr(sound, 'raw_signal'))
        # instantiation from existing signal
        signal = helpers_tests.get_rnd_test_Signal()
        sound = Sound((signal.signal, signal.sr))
        self.assertIsInstance(sound, Sound)
        self.assertTrue(hasattr(sound, 'raw_signal'))

    def test_Sound_condition(self):
        """ Test for the Sound condition method"""
        # Test normal case
        file = helpers_tests.get_rnd_audio_file()
        sound = Sound(file)
        sound.condition()
        self.assertTrue(sound.signal is not None)
        # test return self
        sound2 = Sound(file)
        sound2 = sound2.condition(return_self=True)
        self.assertTrue(sound2.signal is not None)
        # test auto trim
        file = helpers_tests.get_rnd_audio_file()
        sound4 = Sound(file)
        sound4.condition(auto_trim=True)
        self.assertTrue(sound4.signal is not None)

    def test_Sound_use_raw_signal(self):
        """ Test for the use_raw_signal method of the Sound class"""
        # test the regular case
        file = helpers_tests.get_rnd_audio_file()
        sound = Sound(file)
        sound.use_raw_signal()
        self.assertTrue(sound.signal is not None)
        # test with return self
        file = helpers_tests.get_rnd_audio_file()
        sound = Sound(file)
        sound = sound.use_raw_signal(return_self=True)
        self.assertTrue(sound.signal is not None)

    def test_Sound_bin_divide(self):
        """ Test for the bin divide method of the Sound class """
        # This method is already tested in the Signal class tests
        file = helpers_tests.get_rnd_audio_file()
        sound = Sound(file)
        bins = sound.raw_signal.make_freq_bins()
        self.assertIsInstance(bins, dict)

    def test_Sound_trim_signal(self):
        """ Test for the trim signal method of the Sound class """
        file = helpers_tests.get_rnd_audio_file()
        sound = Sound(file)
        sound.trim_signal()
        self.assertTrue(sound.trimmed_signal is not None)
        self.assertTrue(len(sound.trimmed_signal.time()) < len(sound.raw_signal.time()))

    def test_Sound_listen_freq_bins(self):
        """ Test for the listen freq bins method of the Sound class """
        file = helpers_tests.get_rnd_audio_file()
        sound = Sound(file)
        sound.condition()
        sys.stdout = io.StringIO()
        sound.listen_freq_bins()
        output = sys.stdout.getvalue()
        for key in sound.bins:
            self.assertTrue(key in output)
        self.assertTrue('<IPython.lib.display.Audio object>' in output)

    def test_Sound_plot_freq_bins(self):
        """ Test the plot freq bins method of the Plot class"""
        file = helpers_tests.get_rnd_audio_file()
        sound = Sound(file)
        sound.condition()
        sound.plot_freq_bins('all')
        ax = plt.gca()
        self.assertTrue(len(ax.lines) > 0)
        sound.plot_freq_bins(['bass', 'mid'])
        ax = plt.gca()
        self.assertTrue(len(ax.lines) > 0)

    def test_Sound_peak_damping(self):
        """ Test the peak damping method of the Sound class """
        file = helpers_tests.get_rnd_audio_file()
        sound = Sound(file)
        sound.condition()
        sound.peak_damping()
        sys.stdout = io.StringIO()
        sound.listen_freq_bins()
        output = sys.stdout.getvalue()
        self.assertTrue(output != [])

    def test_Sound_bin_hist(self):
        """ Test the bin hist method of the Sound class """
        file = helpers_tests.get_rnd_audio_file()
        sound = Sound(file)
        sound.condition()
        sound.bin_hist()
        ax = plt.gca()
        self.assertTrue(len(ax.patches) > 0)

    def test_Sound_trim_sounds(self):
        files = [helpers_tests.get_rnd_audio_file() for _ in range(3)]
        sounds = []
        for f in files:
            sound = Sound(f)
            sound.condition()
            sounds.append(sound)
        min_len = np.min([s.signal.time()[-1] for s in sounds])
        new_sounds1 = utils.trim_sounds(*sounds)
        for S in new_sounds1:
            self.assertTrue(np.abs(min_len - S.signal.time()[-1]) < 1e-3)
        new_sounds2 = utils.trim_sounds(*sounds, length=1.0)
        for S in new_sounds2:
            self.assertTrue(np.abs(1.0 - S.signal.time()[-1]) < 1e-3)


if __name__ == '__main__':
    unittest.main()
