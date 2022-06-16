import unittest
from guitarsounds import Signal
from guitarsounds.helpers_tests import get_rnd_test_Signal, get_ref_test_Signal
import numpy as np
import os
from contextlib import redirect_stdout
import io


class MyTestCase(unittest.TestCase):

    def test_Signal_instantiation(self):
        """ Test that a Signal class is created from a soundfile"""
        signal = get_rnd_test_Signal()
        self.assertIsInstance(signal, Signal)
        return signal

    def test_Signal_time(self):
        """ Test that the time vector of the signal is correct"""
        signal = get_ref_test_Signal()
        self.assertAlmostEqual(signal.time()[-1], 10.473333333333333)

    def test_Signal_listen(self):
        """ Test the behaviour of the Signal.listen() method"""
        signal = get_rnd_test_Signal()
        f = io.StringIO()
        with redirect_stdout(f):
            signal.listen()
        s = f.getvalue()
        # This is a very wierd test, and it may break with the feature still working
        # A better way to test IPython outputs probably exists
        self.assertEqual(s, '<IPython.lib.display.Audio object>\n')
        # test that the temporary file was removed
        self.assertFalse('temp.wav' in os.listdir())

    def test_Signal_fft(self):
        """ Test the signal fourier transform and corresponding frequency array"""
        signal = get_ref_test_Signal()
        fft = signal.fft()
        fft_freq = signal.fft_frequencies()
        self.assertAlmostEqual(fft[0], 0.00434514742092674)
        self.assertAlmostEqual(fft[-1], 2.106108138485372e-06)
        self.assertAlmostEqual(fft_freq[0], 0.)
        self.assertAlmostEqual(fft_freq[-1], 11024.904519000935)

    def test_Signal_peaks(self):
        """ Test the fourier transform peak analysis methods of the Signal class"""
        # test for random signal
        rnd_sig = get_rnd_test_Signal()
        peak_idxs = rnd_sig.peaks()
        # test that peaks were found
        self.assertTrue(len(peak_idxs) > 0)
        # test for known signal
        ref_sig = get_ref_test_Signal()
        peak_idxs = ref_sig.peaks()
        note_frequency = ref_sig.fft_frequencies()[peak_idxs[0]]
        # Test that the first peak is the G3 note (196 Hz)
        self.assertTrue(abs(196 - note_frequency) < 1)

    def test_Signal_peak_damping(self):
        """ Test the peak damping method of the signal"""
        # test for random signal
        rnd_sig = get_rnd_test_Signal()
        # higher frequency is less damped
        freq_damping = rnd_sig.peak_damping()
        self.assertTrue(freq_damping[-1] < freq_damping[0])
        # test for known signal
        ref_sig = get_ref_test_Signal()
        freq_damping = ref_sig.peak_damping()
        self.assertAlmostEqual(freq_damping[0], 0.001719112228705382)
        self.assertAlmostEqual(freq_damping[-1], 0.00020142136609286606)

    def test_Signal_fundamental(self):
        """ Test the signal fundamental finding method"""
        sig = get_ref_test_Signal()
        fund = sig.fundamental()
        # Fundamental is G3 = 196 Hz
        self.assertTrue(abs(fund - 196) < 1)

    def test_Signal_cavity_frequency(self):
        """ Test the cavity peak finding Signal method"""
        sig = get_ref_test_Signal()
        cavity_freq = sig.cavity_frequency()
        self.assertTrue(cavity_freq > 0.)
        self.assertAlmostEqual(cavity_freq, 106.74775695430769)

    def test_Signal_time_daping(self):
        """ Test the Signal time damping computation"""
        # random signal test
        sig = get_rnd_test_Signal()
        self.assertTrue(sig.time_damping() > 0.)
        # reference signal test
        sig = get_ref_test_Signal()
        self.assertAlmostEqual(sig.time_damping(), 0.00040576327963433924)

    def test_Signal_fft_bins(self):
        """ Test the Signal frequency distribution of the fft"""
        sig = get_ref_test_Signal()
        fft_bins = sig.fft_bins()
        self.assertAlmostEqual(fft_bins[0], 1.9096199812935186)
        self.assertAlmostEqual(fft_bins[-1], 6393.789621366959)

    def test_Signal_envelop(self):
        """ Test the envelop computation of the Signal"""
        sig = get_rnd_test_Signal()
        self.assertAlmostEqual(0, sig.envelop()[0])
        sig = get_ref_test_Signal()
        self.assertAlmostEqual(0.016528025, sig.envelop()[-1])

    def test_Signal_envelop_time(self):
        """ Test the envelop time vector of the signal"""
        sig = get_rnd_test_Signal()
        self.assertTrue(sig.envelop_time()[0] == 0.)
        self.assertTrue(abs(sig.envelop_time()[-1] - sig.time()[-1]) < 1)

    def test_Signal_log_envelop_time(self):
        """ Test the logarithmic time envelop computation"""
        sig = get_rnd_test_Signal()
        log_env, log_time = sig.log_envelop()
        # zero removed from time
        self.assertTrue(log_time[0] > 0)
        self.assertTrue(log_env[-1] < np.max(log_env))

    def test_Signal_find_onset(self):
        """  test the method finding the onset of the signal"""
        sig = get_rnd_test_Signal()
        self.assertTrue(sig.find_onset() < len(sig.time()))

    def test_Signal_trim_onset(self):
        """ Test the method trimming the signal at its onset"""
        sig = get_rnd_test_Signal()
        sig2 = sig.trim_onset()
        # Signal was trimmed ?
        self.assertTrue(sig2.time()[-1] < sig.time()[-1])

    def test_Signal_trim_time(self):
        """ Test the method trimming the time of the Signal"""
        sig = get_rnd_test_Signal()
        sig = sig.trim_time(1.)
        self.assertAlmostEqual(sig.time()[-1], 1.)

    def test_Signal_filter_noise(self):
        """ Possibly remove the filter noise method"""
        sig = get_rnd_test_Signal()
        sig = sig.trim_onset()
        sig2 = sig.filter_noise()
        self.assertIsInstance(sig2, Signal)

    def test_Signal_normalize(self):
        """ Test the method normalizing the amplitude of a signal"""
        sig = get_rnd_test_Signal()
        sig = sig.normalize()
        self.assertAlmostEqual(np.max(np.abs(sig.signal)), 1.)

    def test_Signal_freq_bins(self):
        """ Test the method dividing a signal frequency wise"""
        sig = get_rnd_test_Signal()
        bins = sig.make_freq_bins()
        self.assertIsInstance(bins, dict)
        for k in bins:
            self.assertIsInstance(bins[k], Signal)

    def test_save_signal(self):
        """ Test the Signal saving method """
        sig = get_rnd_test_Signal()
        sig.save_wav('temp')
        self.assertTrue('temp.wav' in os.listdir())
        os.remove('temp.wav')

    def test_spectral_centoid(self):
        """ Test the spectral centroid computation of a signal"""
        sig = get_ref_test_Signal()
        self.assertAlmostEqual(sig.spectral_centroid(), 1309.4894590721722)

if __name__ == '__main__':
    unittest.main()
