import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import curve_fit, interp1d
import dev_utils as du

nylon_sounds = du.load_nylon_sounds()


def strictly_increasing(curve, atol=1e-3):
    diff = curve[1:] - curve[:-1]
    idx = np.arange(diff.shape[0])[np.logical_and(diff < -atol, curve[:-1] > 0.1)][0]
    return idx


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
        return (None, None)
        if plot:
            plt.plot(x[:idx], y[:idx], color='blue')
            
            
def plot_exp(a, b, **kwargs):
    """ Plot exponent curve from a, b """
    t_interp = np.linspace(0, 0.2, 400)
    exp_hi_draft = a * np.exp(t_interp * b)
    hi_interp = interp1d(exp_hi_draft, t_interp)
    off_set = hi_interp(1) - 0.1
    t = np.linspace(0, 0.1, 200)
    exp_hi = a * np.exp((t + off_set) * b)
    plt.plot(t, exp_hi, **kwargs)
    
    
def exp_curve(a, b, start=0, stop=0.1, step=200):
    """ Scaled exponent curve from a, b parameters """
    t_interp = np.linspace(start, stop * 2, step * 2)
    exp_draft = a * np.exp(t_interp * b)
    interp = interp1d(exp_draft, t_interp)
    off_set = interp(1) - 0.1
    t = np.linspace(start, stop, step)
    exp_c = a * np.exp((t + off_set) * b)
    return exp_c
    

nylona = []
nylonb = []
sounds = [nylon_sounds[ky] for ky in nylon_sounds if ky != 'A5']
for s in sounds:
    a, b = get_exp_params(s, plot=False, trim=-1)
    nylona.append(a)
    nylonb.append(b)
    
t = np.linspace(0, 0.1, 200)
plt.fill_between(t, exp_curve(np.min(nylona), np.max(nylonb)), exp_curve(np.max(nylona), np.min(nylonb)), 
                 color='orange', alpha=0.5, label='nylon')
plt.fill_between(t, du.get_low_exp(0, 0.1, 200), du.get_hi_exp(0, 0.1, 200), 
                 color='blue', alpha=0.5, label='other')
    
plt.legend(loc='upper left')
plt.gca().set_xlim(0.08, 0.101)
plt.gca().set_ylim(-0.1, 1.1)
plt.gca().set_xlabel('time (s)')
plt.gca().set_xlabel('amplitude')
plt.gcf().set_size_inches(10, 7)
