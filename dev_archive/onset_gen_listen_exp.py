import dev_utils as du
import numpy as np
import guitarsounds
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

s = du.load_wood_sounds()[0]
s.SP.change('fft_range', 8000)
time_intervals = np.linspace(0.1, 1, 10)
sub_sigs = []
for i, _ in enumerate(time_intervals[:-1]):
    # create a signal from subset
    idx1 = du.time_index(s.signal, time_intervals[i])
    idx2 = du.time_index(s.signal, time_intervals[i + 1])
    new_sig = guitarsounds.Signal(s.signal.signal[idx1:idx2], s.signal.sr, s.SP)
    sub_sigs.append(new_sig)
    
peaks = s.signal.peaks()
peak_freqs = s.signal.fft_frequencies()[peaks]
center_times = [np.mean([time_intervals[i], time_intervals[i + 1]]) for i in range(len(time_intervals) - 1)]

t = s.signal.time()
t = t[t < 1]

# for each peak
new_sig = 0
itrps = []
for i, pf in enumerate(peak_freqs):
    # for each sub signal get the peak amplitude
    amps = []
    for sig in sub_sigs:
        fidx = du.frequency_index(sig, pf)
        amps.append(sig.fft()[fidx])
    # Create an interpolator with the center times
    amp_itrp = interp1d(center_times, amps, kind='quadratic', fill_value='extrapolate')
    itrps.append(amp_itrp)
    new_sig += amp_itrp(t) * np.sin(pf * t * 2 * np.pi)
    
new_sig *= 1 / np.max(new_sig)
idx = np.where(t > 1e-1)[0][0]
print('Original')
s.signal.normalize().trim_time(1).listen()

for p in np.linspace(-3, 3, 3):
    onset_exp = du.get_expenv(p)
    onset_env = onset_exp(t[:idx])
    current_sig = new_sig.copy()
    current_sig[:idx] = onset_env * current_sig[:idx]    
    print(f'p = {p}')
    current_sig *= 0.8
    s_current = du.arr2sig(current_sig)
    s_current.plot.signal(label=f'p = {p}')
    s_current.listen()
plt.gca().set_xlim(0.07, 0.11)
plt.legend()
plt.show()
