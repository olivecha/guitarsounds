import DEV_utils as du
import matplotlib.pyplot as plt
import numpy as np

all_sounds = du.low_wood_sounds() + du.load_carbon_sounds()

for i, s in enumerate(all_sounds):
    if i not in bad_sounds:
        s.signal.normalize().plot.signal(color='0.7')
    
b_values = np.array(b_values)
a_values = np.array(a_values)
b_values = b_values[a_values<0.00001]
a_values = a_values[a_values<0.00001]

# Compute the low exponent curve
t_interp = np.linspace(0, 0.15, 200)
exp_low_draft = np.min(a_values)*np.exp(t_interp*np.max(b_values))
low_interp = interp1d(exp_low_draft, t_interp)
off_set = low_interp(1) - 0.1
t = np.linspace(0, 0.1, 200)
exp_low = np.min(a_values)*np.exp((t+off_set)*np.max(b_values))
# Compute the high exponent curve
t_interp = np.linspace(0, 0.12, 200)
exp_hi_draft = np.max(a_values)*np.exp(t_interp*np.min(b_values))
hi_interp = interp1d(exp_hi_draft, t_interp)
off_set = hi_interp(1) - 0.1
t = np.linspace(0, 0.1, 200)
exp_hi = np.max(a_values)*np.exp((t+off_set)*np.min(b_values))
plt.plot(t, exp_low, color='#0049AD', label='onset lower bound')
plt.plot(t, exp_hi, color='#00A0AD', label='onset higher bound')
plt.plot(0, 0, color='0.7', label='all sounds')
plt.fill_between(t, exp_low, exp_hi, alpha=0.3, label='Design space')

plt.gcf().set_size_inches(10, 8)
plt.gca().set_xlim(0.08, 0.11)
plt.gca().set_ylim(-0.3, 1.1)
plt.title('Onset bounds for all reference sounds')
plt.legend()
plt.show()