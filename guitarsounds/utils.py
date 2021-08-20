import numpy as np
import scipy.optimize
import scipy.integrate
import os
import ipywidgets as ui


def nth_order_polynomial_residual(A, n, x, y):
    """
    Function computing the residual from a nth order polynomial, with x and y vectors
    :return:
    """
    y_model = 0
    for i in np.arange(n):
        y_model += A[i] * np.array(x) ** i
    return y_model - y


def nth_order_polynomial_fit(n, x, y):
    """
    Function creating a function of a fitted nth order polynomial to a set of x and y data
    :param n: polynomial order
    :param x: x values
    :param y: y values
    :return: the Function such as y = F(x)
    """
    n += 1
    while n > len(x):
        n -= 1
    guess = np.ones(n)
    result = scipy.optimize.least_squares(nth_order_polynomial_residual, guess, args=(n, x, y))
    A = result.x

    def polynomial_function(x):
        y = 0
        for i, a in enumerate(A):
            y += a * np.array(x) ** i
        return y

    return polynomial_function


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


def power_split(y, x, x_max, n):
    """
    Computes the indexes corresponding to equal integral values for a function y and x values
    :param y: function value array
    :param x: function x values array
    :param x_max: maximum x value
    :param n: number of integral bins
    :return: indexes of the integral bins
    """
    imax = np.nonzero(x > x_max)[0]
    A = scipy.integrate.trapezoid(y[:imax])
    I = A / n
    indexes = [0] * n
    for i in range(n - 1):
        i1 = indexes[i]
        i2 = i1 + 1
        while scipy.integrate.trapezoid(y[i1:i2]) < I:
            i2 += 1
        indexes[i + 1] = i2
    return indexes
