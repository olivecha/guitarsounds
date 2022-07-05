import guitarsounds
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('\..')
import DEV_utils as du

# Finding a viable interpolator for a(f) amplitude frequency function
du.bigfig()
s.SP.change('fft_range', 10000)
#sig.plot.fft(label=r'$\sum_{n=0}^N a(n) sin(2 \pi f t)$')
s.signal.plot.fft(label='signal FFT', alpha=0.6, color='0.4')
plt.gca().set_xlim(0, 10000)
#plt.plot(s.signal.fft_frequencies(), a0(s.signal.fft_frequencies()))
#plt.yscale('linear')
freqs = s.signal.fft_frequencies()
a1 = np.exp((-10/6000) * freqs)
plt.plot(freqs, a1, label = r'$e^{-f}$', color='green')

a2 = scipy.interpolate.InterpolatedUnivariateSpline(pfreqs, pamps, k=2)
plt.plot(freqs, a2(freqs), label='spline', color='orange')

a3 = scipy.interpolate.interp1d(pfreqs, pamps, bounds_error=False, fill_value=(pamps[0], pamps[-1]))
plt.plot(freqs, a3(freqs), label='linear', color='red')
plt.title(r'Interpolators to be used to approximate $a_n(f)$ inside a frequency bin')
plt.legend()
plt.show()

# Signal reconstruction with frequency peak value interpolation
sp = guitarsounds.parameters.sound_parameters()

fbins = {'bass':sp.bins.bass.value,
         'mid':sp.bins.mid.value,
         'highmid':sp.bins.highmid.value,
         'uppermid':sp.bins.uppermid.value,
         'presence':sp.bins.presence.value, 
         'brillance':10000}

pfreqs = s.signal.fft_frequencies()[s.signal.peaks()]
pfreqs = np.append(pfreqs, pfreqs[-1] + s.fundamental)
pamps = s.signal.fft()[s.signal.peaks()]
pamps = np.append(pamps, pamps[-2])
a0 = scipy.interpolate.interp1d(pfreqs, pamps, bounds_error=False, fill_value=(pamps[0], pamps[-1]))
freq = s.fundamental
time = s.signal.time()
time = time[time<1]
envelops = []
sig = 0
freqs = []
while freq < fbins['brillance']:
    freqs.append(freq)
    freq += s.fundamental
freqs.append( freq+ s.fundamental)
an_values = a0(freqs)
an_values /= np.sum(an_values)

sig = 0
i = 0
for b in s.bins:
    mf = fbins[b]
    env, tim = s.bins[b].normalize().log_envelop()
    bin_env = interp1d(tim, env)
    sig_e = 0
    while freqs[i] < mf:
        sig_e +=  an_values[i] * np.sin(freqs[i]*time*2*np.pi)
        i += 1
    sig += bin_env(time) * sig_e

sig /= np.max(sig)
sig = du.arr2sig(sig)
# Signal plot for original and reconstructed
sig.SP.change('fft_range', 10000)
sig.listen()
sig.plot.signal()
s.signal.plot.signal()
# Fourier transform plot for original and reconstructed signals
plt.figure()
plt.gca().set_xlim(0, 1)
sig.plot.fft(alpha=0.6)
s.signal.plot.fft(alpha=0.6)

# Frequency bins envelop plot for reconstructed and original signal
plt.figure()
du.bigfig()
bins = sig.make_freq_bins()
gen_cm = plt.colormaps['viridis']
colorange = np.linspace(0, 0.8, 6)
for i, b in enumerate(bins):
    bins[b].normalize().plot.log_envelop(color=gen_cm(colorange[i]), alpha=0.6)
for i, b in enumerate(s.bins):
    s.bins[b].trim_time(1).normalize().plot.log_envelop(linestyle='--',color=gen_cm(colorange[i]), alpha=0.6)
plt.plot(0, 0, color='0.3', label='reconstructed')
plt.plot(0, 0, '--',color='0.3', label='real signal')
plt.legend()
plt.title('Frequency bins envelops of the original and reconstructed signal')
plt.show()

