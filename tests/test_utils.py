import unittest
import filecmp
from guitarsounds import utils, Sound, Signal
from guitarsounds.parameters import sound_parameters
from helpers_tests import get_rnd_test_Signal, get_ref_test_Signal
from helpers_tests import get_rnd_audio_file
import numpy as np
import os
import io


class MyTestCase(unittest.TestCase):
    """ pytest class for guitarsounds.utils submodule """

    def test_polynomial_residual(self):
        """ Test the polynomial residual function """
        # polynomial coefficients
        coefficients = [1, 2, 6, -4]

        x_values = np.linspace(-10, 10)
        y_values = np.zeros_like(x_values)
        for order, coeff in enumerate(coefficients):
            y_values += coeff * x_values ** order

        residual = utils.nth_order_polynomial_residual(coefficients,
                                                       len(coefficients),
                                                       x_values,
                                                       y_values)
        self.assertAlmostEqual(0., np.sum(residual))

    def test_polynomial_fit(self):
        """
        Test the polynomial fitting function
        by fitting a known polynomial
        """
        x_values = np.linspace(-10, 10)
        y_values = -x_values**4 + 3*x_values**2 - 2 * x_values + 15
        poly_fun = utils.nth_order_polynomial_fit(4, x_values, y_values)
        y_eval = poly_fun(x_values)
        self.assertAlmostEqual(0., np.sqrt(np.sum((y_eval - y_values)**2)))

    def test_octave_values(self):
        """
        Test the octave value function
        """
        self.assertAlmostEqual(11.0485435, utils.octave_values(2)[0], 3)
        self.assertAlmostEqual(12.4015707, utils.octave_values(3)[0], 3)

    def test_octave_histogram(self):
        """
        test the octave histogram function
        """
        self.assertAlmostEqual(9.2906805, utils.octave_histogram(2)[0], 3)
        self.assertAlmostEqual(11.0485435, utils.octave_histogram(3)[0], 3)

    def test_trim_sounds(self):
        """
        Test the trim_sounds utils.py function
        """
        # Get random audio files
        files = []
        for i in range(3):
            files.append(get_rnd_audio_file())
        sounds = [Sound(file).condition(return_self=True) for file in files]
        # Branch with length = None 
        old_times = [si.signal.time() for si in sounds]
        new_sounds = utils.trim_sounds(*sounds)
        new_times = [si.signal.time() for si in new_sounds]
        min_time = np.min([ti[-1] for ti in old_times])
        for ti in new_times:
            self.assertAlmostEqual(ti[-1], min_time, 3)
        # Branch with specified length
        new_sounds = utils.trim_sounds(*sounds, length=1.0)
        for si in new_sounds:
            self.assertAlmostEqual(1.0, si.signal.time()[-1], 3)

    def test_load_wav(self):
        """ test the load_wav function from utils.py """
        # Load a random wav file
        wav_file = get_rnd_audio_file()
        sig, sr = utils.load_wav(wav_file)
        # Create a signal object
        test_signal = Signal(sig, sr, sound_parameters())
        # save it to a wav_file and check for preservation
        test_signal.save_wav('temp')
        self.assertTrue(filecmp.cmp(wav_file, 'temp.wav'))
        os.remove('temp.wav')

    def test_resample(self):
        """ Test the resample function in utils.py """
        test_signal = get_rnd_test_Signal()
        # resample to 10000
        rsp_sig = utils.resample(test_signal.signal, test_signal.sr, 10000)
        final_time = len(rsp_sig)/10000
        self.assertAlmostEqual(test_signal.time()[-1], final_time, 3)


