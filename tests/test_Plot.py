import unittest
from guitarsounds import Plot
from guitarsounds import helpers_tests
import matplotlib.pyplot as plt

class MyTestCase(unittest.TestCase):
    """ Test for the poltting class"""

    def test_Plot_instantiation(self):
        """ Test that the class can be instantiated"""
        plot = Plot()
        self.assertIsInstance(plot, Plot)

    def test_Plot_sanitize_kwargs(self):
        """ Test the sanitize_kwargs method of the Plot class"""
        plot = Plot()
        illegal_kwargs = ['max_time', 'n', 'ticks', 'normalize', 'inverse', 'peak_height', 'fill']
        kwargs = plot.sanitize_kwargs(illegal_kwargs)
        self.assertEqual(kwargs, {})

    def test_Plot_set_bin_ticks(self):
        """ Tests the set_bin_ticks method of Plot class"""
        parent = helpers_tests.get_rnd_test_Signal()
        plot = Plot()
        plot.parent = parent
        plot.fft()
        plot.set_bin_ticks()
        ax = plt.gca()
        # Highly dependent on the deep matplotlib API
        self.assertIsInstance(ax.xaxis.major.formatter.seq[0], str)

    def test_Plot_signal(self):
        """ Test the signal Plot method"""
        parent = helpers_tests.get_rnd_test_Signal()
        parent.plot.signal()
        ax = plt.gca()
        self.assertTrue(hasattr(ax, 'lines'))

    def test_Plot_envelop(self):
        """ Test the envelop Plot method"""
        parent = helpers_tests.get_rnd_test_Signal()
        parent.plot.envelop()
        ax = plt.gca()
        self.assertTrue(hasattr(ax, 'lines'))

    def test_Plot_log_envelop(self):
        """ Test the log envelop Plot method"""
        parent = helpers_tests.get_rnd_test_Signal()
        parent.plot.log_envelop()
        ax = plt.gca()
        self.assertTrue(len(ax.lines) > 0)

    def test_Plot_fft(self):
        """ Test the fft Plot method """
        parent = helpers_tests.get_rnd_test_Signal()
        parent.plot.fft()
        ax = plt.gca()
        # Check that the frequency range was set
        self.assertTrue(abs(ax.dataLim.extents[2] - parent.SP.general.fft_range.value) < 1)
        self.assertTrue(len(ax.lines) > 0)

    def test_Plot_fft_hist(self):
        """ Tests the fft hist Plot method """
        parent = helpers_tests.get_rnd_test_Signal()
        parent.plot.fft_hist()
        ax = plt.gca()
        self.assertTrue(len(ax.patches) > 0)

    def test_Plot_peaks(self):
        """ Test the peak Plot method"""
        parent = helpers_tests.get_rnd_test_Signal()
        parent.plot.peaks()
        ax = plt.gca()
        self.assertTrue(len(ax.lines) > 0)

    def test_Plot_peak_damping(self):
        """ Test the peak damping Plot method"""
        parent = helpers_tests.get_rnd_test_Signal()
        parent.plot.peak_damping()
        ax = plt.gca()
        self.assertTrue(len(ax.lines) > 0)

    def test_Plot_time_damping(self):
        """ Test the time damping Plot method """
        parent = helpers_tests.get_rnd_test_Signal()
        parent.plot.time_damping()
        ax = plt.gca()
        self.assertTrue(len(ax.lines) > 0)

    def test_Plot_integral(self):
        """ Test the integral Plot method """
        parent = helpers_tests.get_rnd_test_Signal()
        parent.plot.integral()
        ax = plt.gca()
        self.assertTrue(len(ax.lines) > 0)


if __name__ == '__main__':
    unittest.main()
