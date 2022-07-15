import DEV_utils as du
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import curve_fit


all_sounds = du.load_wood_sounds() + du.load_carbon_sounds()
du.bigfig()


def strictly_increasing(curve, atol=1e-3):
    diff = curve[1:] - curve[:-1]
    idx = np.arange(diff.shape[0])[np.logical_and(diff < -atol, curve[:-1] > 0.1)][0]
    return idx


def plot_exp(a, b, **kwargs):
    def new_exp(x):
        return a * np.exp(x * b)
    x = np.linspace(0, 0.1)
    y = new_exp(x)
    plt.plot(x, y, **kwargs)


def get_exp_params(s, trim=0, plot=False, **kwargs):
    """ Get the fitted onset exponential parameters from a sound class """
    x = s.signal.normalize().envelop_time(51, 25)
    y = s.signal.normalize().envelop(51, 25)
    # Use only the strictly increasing part
    idx = strictly_increasing(y) - trim
    if x[idx] < 0.091:
        if plot:
            plt.plot(x[:idx], y[:idx], color='orange')
            s.signal.plot.signal(color='orange')
        raise ValueError('Something is wrong with the signal')
    x = x[:idx]
    y = y[:idx]
    if plot:
        plt.plot(x, y, color='#AD0034', alpha=0.3)

    def env_exp(x, a, b):
        return a * np.exp(x * b)
    try:
        (a, b), _ = curve_fit(env_exp, x, y, **kwargs)
        return a, b
    except RuntimeError:
        if plot:
            plt.plot(x[:idx], y[:idx], color='orange')


a_values, b_values = [], []
bad_sounds = []

for i, s in enumerate(all_sounds):
    try:
        a, b = get_exp_params(s, plot=True, maxfev=1000)
        plt.plot(s.signal.time(), s.signal.normalize().signal, color='0.5', alpha=0.3)

    except:
        bad_sounds.append(i)
        pass

    if a < 0.00001:
        plot_exp(a, b, color='blue', alpha=0.3)

    a_values.append(a)
    b_values.append(b)
    
    
plt.plot(0, 0, color='0.5', label='all signals')
plt.plot(0, 0, color='blue', label='onset fit')
plt.plot(0, 0, color='#AD0034', label='onset real')
plt.plot(0, 0, color='orange', label='rejected signals')

plt.legend()
plt.title('Onset fit for all reference signals')
plt.gcf().set_size_inches(10, 8)
plt.gca().set_xlim(0.07, 0.11)
plt.gca().set_ylim(-0.3, 1.1)
plt.show()
