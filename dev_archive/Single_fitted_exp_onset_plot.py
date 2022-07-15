""" docstring """
import sys
import DEV_utils as du
sys.path.append('/..')
sys.path.append('/../..')
import matplotlib.pyplot as plt
from scipy.interpolate import curve_fit
import numpy as np

du.bigfig()
all_sounds = du.load_wood_sounds()
s = all_sounds[4]
s.signal.normalize().plot.signal()
ax = plt.gca()
ax.set_xlim(0.05, 0.15)
onset = s.signal.find_onset()
y, x = s.signal.normalize().log_envelop()
y = y[x <= 0.101]
x = x[x <= 0.101]


def env_exp(x, a, b):
    """ doc """
    return a * np.exp(x * b)


(a, b), _ = curve_fit(env_exp, x, y)


def env(t):
    return a * np.exp(b * t)


sys.path.append('/..')
env_time = np.linspace(0, 0.1)
plt.plot(env_time, env(env_time), label='fitted curve')
plt.plot(x, y, label='log envelop')
plt.scatter(s.signal.time()[onset], s.signal.normalize().signal[onset], marker='x', color='blue')
plt.legend()
plt.title('Fitted exponential and log envelop for a single signal')
plt.show()
