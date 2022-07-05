import guitarsounds
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/..')
import DEV_utils as du

############################################
# Signal envelop shape interpolation
############################################
wood_sounds = du.load_wood_sounds()
s = wood_sounds[1]
plt.figure(figsize=(8, 5))
s.signal.normalize().plot.log_envelop(color='k',alpha=0.5)
s.signal.normalize().plot.signal()
lenv, ltime = s.signal.normalize().log_envelop()
#plt.scatter(ltime, lenv, 10)
plt.gca().set_xlim(7e-2, 25e-1)

# Onset interpolation
idx_end = np.arange(ltime.size)[ltime>0.1][0]
idx_start = idx_end - 2
plt.scatter(ltime[idx_start:idx_end+1], lenv[idx_start:idx_end+1], 20, marker='x', color='red')
def onset_exp(t, a, b):
    return a*np.exp(b*t)
(a, b), _ = curve_fit(onset_exp, ltime[idx_start:idx_end+1], lenv[idx_start:idx_end+1])
onset_time = np.linspace(ltime[idx_start]/2, ltime[idx_end])
plt.plot(onset_time, onset_exp(onset_time, a, b), color='red')
#plt.title(r'First part of the interpolated envelop, $params=(a, b)$')
plt.annotate(r'$env = ae^{bt}$', (8e-2, 0.3), fontsize=13)

# decay interpolation
idx_start = idx_end
idx_end = np.arange(ltime.size)[ltime>3][0]
plt.scatter(ltime[idx_start:idx_end+1], lenv[idx_start:idx_end+1],20, marker='x', color='blue')
sigma=np.linspace(0.0001, 0.001, ltime[idx_start:idx_end+1].shape[0])

def decay_poly(x, a, b, c, d, e, f, g, h, i, j):
    y = j*x**9 + i*x**8 + h * x **7 + g * x**6 + f * x **5 + e * x ** 4 + a * x**3 + b * x**2 + c * x  + d 
    return y
coefs, _ = curve_fit(decay_poly, ltime[idx_start:idx_end+1], lenv[idx_start:idx_end+1], sigma=sigma)
decay_time = np.linspace(ltime[idx_start], 2)
plt.plot(decay_time, decay_poly(decay_time, *coefs), color='blue')
plt.title(r'Second part of the interpolated envelop, $params=(a, b, c, d, e, f, g, h)$')
plt.annotate(r'$env = ax^7 + bx^6 + cx^5 + dx^4 +ex^3 + fx^2 + gx + h$', (14e-2, 0.8), fontsize=13)

peak_freqs = s.signal.fft_frequencies()[s.signal.peaks()]
peak_amps = s.signal.fft()[s.signal.peaks()]
peak_decay = s.signal.peak_damping()
peak_decay = 1/peak_decay
peak_decay = peak_decay/np.max(peak_decay)
amps = peak_amps / np.sum(peak_amps)
t = s.signal.time()
t = t[t<3]
new_s = 0

idx = np.where(t>1e-1)[0][0]
onset_env = onset_exp(t[:idx], a, b)
decay_env = decay_poly(t[idx:], *coefs)

for f, amp, d in zip(peak_freqs, amps, peak_decay):
    si = amp*np.sin(f*t*2*np.pi)
    si[:idx] = onset_env*si[:idx]
    si[idx:] = decay_env*si[idx:]*d*50
    new_s += si

new_s *= 1/np.max(new_s)

print('Reconstructed signal')
sig = guitarsounds.Signal(new_s, s.signal.sr, guitarsounds.parameters.sound_parameters())
sig.listen()
print('Original signal')
s.signal.normalize().listen()