import numpy as np
import scipy.optimize

def nth_order_polynomial_residual(A, n, x, y):
    y_model = 0
    for i in np.arange(n):
        y_model += A[i] * np.array(x) ** i
    return y_model - y


def nth_order_polynomial_fit(n, x, y):
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