import DEV_utils as du
import numpy as np
import matplotlib.pyplot as plt

# function should take a value between 0 and 1
# hard coded values : 
a_min = 1.216110966595552e-21
a_max = 2.956038122585772e-11
b_min = 241.88870914170292
b_max = 476.8504184761331

# 0 is the lowest bound and 1 the highest
fin = 0.5
# a_max is the highest curve
a = a_min + fin*(a_max - a_min)
# b_min is the highest curve
b = b_min + (1 - fin)*(b_max - b_min)

# Correct the value at t = 0.1 at run time
t1 = np.linspace(0, 0.15, 200)
env_exp_draft = a * np.exp(t1 * b)
itrp = interp1d(env_exp_draft, t1)
offset = itrp(1) - 0.1
# Create the callable for the current envelop value
def expenv(t):
    return a*np.exp((t+offset)*b)

# Compute the low exponent curve
t_interp = np.linspace(0, 0.15, 200)
exp_low_draft = a_min*np.exp(t_interp*b_max)
low_interp = interp1d(exp_low_draft, t_interp)
off_set = low_interp(1) - 0.1
t = np.linspace(0, 0.1, 200)
exp_low = a_min*np.exp((t+off_set)*b_max)

# Compute the high exponent curve
t_interp = np.linspace(0, 0.12, 200)
exp_hi_draft = a_max*np.exp(t_interp*b_min)
hi_interp = interp1d(exp_hi_draft, t_interp)
off_set = hi_interp(1) - 0.1
t = np.linspace(0, 0.1, 200)
exp_hi = a_max*np.exp((t+off_set)*b_min)


t = np.linspace(0, 0.1, 200)

du.bigfig()
plt.plot(t, exp_low, color='#0049AD', label='lower bound')
plt.plot(t, exp_hi, color='#00A0AD', label='higher bound')
plt.plot(t, expenv(t), color='orange', label='p = 0.5')
plt.gca().set_xlim(0.08, 0.11)
plt.legend()
plt.title('Max min and intermediary onset value')