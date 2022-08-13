import dev_utils as du
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# Load all the sounds
wood_sounds = du.load_wood_sounds()
carbon_sounds = du.load_carbon_sounds()
nylon_sounds = du.load_nylon_sounds()
all_sounds = wood_sounds + carbon_sounds + nylon_sounds
notes = du.get_notes_names()
wood_sounds = du.arr2notedict(wood_sounds)
carbon_sounds = du.arr2notedict(carbon_sounds)
nylon_sounds = du.arr2notedict(nylon_sounds)

# Get all the sounds and their fundamentals
sorted_sounds = []
sorted_notes = []
for n in wood_sounds:
    sorted_sounds.append(wood_sounds[n])
    sorted_notes.append(n)
for n in carbon_sounds:
    sorted_sounds.append(carbon_sounds[n])
    sorted_notes.append(n)
for n in nylon_sounds:
    sorted_sounds.append(nylon_sounds[n])
    sorted_notes.append(n)
fundamentals = np.array([110, 247, 147, 330, 82, 196] * 3)
fundamentals[6:] += 5
fundamentals[12:] += 5
fundamentals_n = fundamentals / np.max(fundamentals)

fig, axs = plt.subplots(1, 2, figsize=(18, 6))

# FIRST PLOT
plt.sca(axs[0])

fft_range_values = np.linspace(2000, 14000, 10)
mymap = plt.get_cmap('viridis')

for s, fn in zip(sorted_sounds, fundamentals_n):
    peak_n = []
    for ftran in fft_range_values:
        s.signal.SP.change('fft_range', ftran)
        try:
            peaks = s.signal.peaks_old()
        except IndexError:
            peaks = []
        peak_n.append(len(peaks))

    plt.plot(fft_range_values, peak_n, color=mymap(fn))
    
# Create a color map from the plot lines
norm = mpl.colors.Normalize(vmin=np.min(fundamentals), vmax=np.max(fundamentals))
sm = plt.cm.ScalarMappable(cmap=plt.colormaps['viridis'], norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ticks=fundamentals[:6])
cbar.set_label('Sound fundamental')
    
# Plot setup
ax = axs[0]
ax.set_xlabel('Allowed frequeny range (Hz)')
ax.set_ylabel('Number of extracted peaks')
ax.set_title('Whole frequency range')

# SECOND PLOT
plt.sca(axs[1])

fft_range_values = np.linspace(10000, 11500, 10)
mymap = plt.get_cmap('viridis')

for s, fn in zip(sorted_sounds, fundamentals_n):
    peak_n = []
    for ftran in fft_range_values:
        s.signal.SP.change('fft_range', ftran)
        try:
            peaks = s.signal.peaks_old()
        except IndexError:
            peaks = []
        peak_n.append(len(peaks))

    plt.plot(fft_range_values, peak_n, color=mymap(fn))
    
# Create a color map from the plot lines
norm = mpl.colors.Normalize(vmin=np.min(fundamentals), vmax=np.max(fundamentals))
sm = plt.cm.ScalarMappable(cmap=plt.colormaps['viridis'], norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ticks=fundamentals[:6])
cbar.set_label('Sound fundamental')
    
# Plot setup
ax = axs[1]
ax.set_xlabel('Allowed frequeny range (Hz)')
ax.set_ylabel('Number of extracted peaks')
ax.set_title('Maximum peak extraction zone')


print('')
print(s.signal.sr, 'samples per second')
print('')
print('Max representable frequency : ', s.signal.fft_frequencies()[-1], 'Hz')

# Get all the sounds and their fundamentals
sorted_sounds = []
sorted_notes = []
for n in wood_sounds:
    sorted_sounds.append(wood_sounds[n])
    sorted_notes.append(n)
for n in carbon_sounds:
    sorted_sounds.append(carbon_sounds[n])
    sorted_notes.append(n)
for n in nylon_sounds:
    sorted_sounds.append(nylon_sounds[n])
    sorted_notes.append(n)
fundamentals = np.array([110, 247, 147, 330, 82, 196] * 3)
fundamentals[6:] += 5
fundamentals[12:] += 5
fundamentals_n = fundamentals / np.max(fundamentals)


du.bigfig()

fft_range_values = np.linspace(1000, 12500, 30)
mymap = plt.get_cmap('viridis')

for s, fn in zip(sorted_sounds, fundamentals_n):
    peak_n = []
    for ftran in fft_range_values:
        s.signal.SP.change('fft_range', ftran)
        try:
            peaks = s.signal.peaks()
        except IndexError:
            peaks = []
        peak_n.append(len(peaks))

    plt.plot(fft_range_values, peak_n, color=mymap(fn))

# Create a color map from the plot lines
norm = mpl.colors.Normalize(vmin=np.min(fundamentals), vmax=np.max(fundamentals))
sm = plt.cm.ScalarMappable(cmap=plt.colormaps['viridis'], norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ticks=fundamentals[:6])
cbar.set_label('Sound fundamental')

# Plot setup
ax = plt.gca()
ax.set_xlabel('Allowed frequeny range (Hz)')
ax.set_ylabel('Number of extracted peaks')
ax.set_title('Corrected peak extraction')

