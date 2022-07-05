import guitarsounds
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('\..')
import DEV_utils as du

# Importing a sound file
s = du.load_wood_sounds(as_dict=True)['A5']
s.signal = s.signal.normalize()
s.signal.listen()

# Visualize the log envelops for each bin
du.bigfig()
for b in s.bins:
    s.bins[b].plot.log_envelop(label=b)
plt.legend()
plt.yscale('log')

# Visualize the FFT for each bin       
du.bigfig()
for b in s.bins:
    s.bins[b].plot.fft(label=b, alpha=0.5, color=None)
plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.xlim(80, 10**4)

# Create an interpolator for the log-envelop
s.mid.plot.log_envelop(label='signal')
env, t  = s.mid.log_envelop()
itrp = interp1d(t, env, kind='linear')
t2 = np.linspace(0, 1, 100)
plt.plot(t2, itrp(t2), color='r', label='interpolator')
plt.legend()
plt.title('Interpolated mid frequenct bin envelop')
plt.show()

# Rebuild a signal such as a sum of sin with the amplitude of the log envelop
sp = guitarsounds.parameters.sound_parameters()

fbins = {'bass':sp.bins.bass.value,
         'mid':sp.bins.mid.value,
         'highmid':sp.bins.highmid.value,
         'uppermid':sp.bins.uppermid.value,
         'presence':sp.bins.presence.value, 
         'brillance':10000}

freq = s.fundamental
time = s.signal.time()
time = time[time<1]
n = 1
sig = 0
for b in s.bins:
    mf = fbins[b]
    env, tim = s.bins[b].log_envelop()
    a = interp1d(tim, env)
    while freq < mf:
        sig += a(time) * np.sin(freq*time*2*np.pi)
        n += 1
        freq += s.fundamental
        
sig /= np.max(sig)

# Listen and plot the the result
DEV_utils.listen_sig_array(sig)
plt.plot(time, sig, label='reconstruction')
s.signal.plot.signal(label='original')
plt.legend()
plt.xlim(0, 1)

# Visulaize the fourier transform of the rebuilt signal
s2 = DEV_utils.arr2sig(sig)
du.bigfig()
s2.SP.change('fft_range', 10000)
s2.plot.fft(label='reconstructed', alpha=0.7)
s.plot.fft(label='original', alpha=0.9)
plt.gca().set_xlim(0, 5000)
plt.legend()
plt.title('Fourier transform of the frequency bin reconstructed signal')


