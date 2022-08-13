import matplotlib.pyplt as plt
import numpy as np
import dev_utils as du
import matplotlib as mpl

fig, axs = plt.subplots(1, 2, figsize=(18, 6))
exp_low = du.get_low_exp(0, 0.1, 200)
exp_hi = du.get_hi_exp(0, 0.1, 200)

plt.sca(axs[0])
p_values = np.linspace(-1, 1, 7)
p_norm = p_values - np.min(p_values)
p_norm /= np.max(p_norm)
t = np.linspace(0, 0.1, 200)
cm = plt.colormaps['Spectral']

for i, p in enumerate(p_values):
    expenv = du.get_expenv(p)
    plt.plot(t, expenv(t), color=cm(p_norm[i]))
plt.fill_between(t, exp_low, exp_hi, color='0.7', alpha=0.3, label='experimental range')
expenv0 = du.get_expenv(0)
plt.plot(t, expenv0(t), color='k', linestyle='--', label='p = 0')
norm = mpl.colors.Normalize(vmin=p_values[0], vmax=p_values[-1])
sm = plt.cm.ScalarMappable(cmap=plt.colormaps['Spectral'], norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ticks=np.linspace(p_values[0], p_values[-1], 7))
cbar.set_label('onset adimensional parameter')
ax = plt.gca()
ax.set_xlabel('time (s)')
ax.set_ylabel('signal')
ax.set_title('Inbounded interpolated envelop generation, p = [-1, 1]')
plt.legend()
plt.gca().set_xlim(0.085, 0.102)

plt.sca(axs[1])
p_values = np.linspace(-4, 4, 15)
p_norm = p_values - np.min(p_values)
p_norm /= np.max(p_norm)
t = np.linspace(0, 0.1, 200)
cm = plt.colormaps['Spectral']

for i, p in enumerate(p_values):
    expenv = du.get_expenv(p)
    plt.plot(t, expenv(t), color=cm(p_norm[i]))
    
plt.fill_between(t, exp_low, exp_hi, color='0.7', alpha=0.3, label='experimental range')
expenv0 = du.get_expenv(0)
plt.plot(t, expenv0(t), color='k', linestyle='--', label='p = 0')
norm = mpl.colors.Normalize(vmin=p_values[0], vmax=p_values[-1])
sm = plt.cm.ScalarMappable(cmap=plt.colormaps['Spectral'], norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ticks=np.linspace(p_values[0], p_values[-1], 7))
cbar.set_label('onset adimensional parameter p')
ax = plt.gca()
ax.set_xlabel('time (s)')
ax.set_ylabel('signal')
ax.set_title('Extrapolated envelop generation, p = [-4, 4]')
plt.legend()
plt.gca().set_xlim(0.085, 0.102)
plt.show()
