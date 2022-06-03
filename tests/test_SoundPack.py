import unittest
import guitarsounds
from guitarsounds import SoundPack, helpers_tests
import numpy as np
import matplotlib.pyplot as plt


class MyTestCase(unittest.TestCase):
    """ Test class for the guitarsounds.Sound class"""

    def test_SoundPack_instantiation_from_files(self):
        """ Test the the SoundPack constructor using files """
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
        pass