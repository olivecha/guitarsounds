import unittest
from guitarsounds import utils
from helpers_tests import get_rnd_test_Signal, get_ref_test_Signal
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
