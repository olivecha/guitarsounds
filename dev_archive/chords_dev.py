import dev_utils as du
import numpy as np
import matplotlib.pyplot as plt

# load wood and nylon sounds
wood_sounds = du.load_wood_sounds()
nylon_sounds = du.load_nylon_sounds()
notes = du.get_notes_names()
wood_sounds = du.arr2notedict(wood_sounds)
nylon_sounds = du.arr2notedict(nylon_sounds)

# Wood sounds chord (arpegio)
chord_sounds = [wood_sounds[ky] for ky in ['E6', 'A5', 'D4', 'G3', 'B2', 'E1']]
arrays = [du.sigarr_gen(s) for s in chord_sounds]
strum_interval = 0.25
s = chord_sounds[-1]
time_idx_step = np.arange(s.signal.time().shape[0])[s.signal.time() > strum_interval][0]
chord = np.zeros(arrays[0].shape[0] + 6 * time_idx_step)
i = 0
for arr in arrays:
    chord[i:i + arr.shape[0]] += arr
    i += time_idx_step

chord *= 0.95 / np.max(np.abs(chord))
plt.plot(chord)
du.listen_sig_array(chord)

# Nylon sounds chord (arpegio)
chord_sounds = [nylon_sounds[ky] for ky in ['E6', 'A5', 'D4', 'G3', 'B2', 'E1']]
max_time = np.min([s.signal.time()[-1] for s in chord_sounds]) - 0.1
arrays = [du.sigarr_gen(s, envelop_param=-3, max_time=max_time) for s in chord_sounds]
strum_interval = 0.19
time_idx_step = np.arange(s.signal.time().shape[0])[s.signal.time() > strum_interval][0]
chord = np.zeros(arrays[0].shape[0] + 6 * time_idx_step)
i = 0
for arr in arrays:
    chord[i:i + arr.shape[0]] += arr
    i += time_idx_step

chord *= 0.95 / np.max(np.abs(chord))
plt.plot(chord)
du.listen_sig_array(chord)

# Nylon sounds chord (strummed)
chord_sounds = [nylon_sounds[ky] for ky in ['E6', 'A5', 'D4', 'G3', 'B2', 'E1']]
max_time = np.min([s.signal.time()[-1] for s in chord_sounds]) - 0.1
arrays = [du.sigarr_gen(s, envelop_param=-3, max_time=max_time) for s in chord_sounds]
strum_interval = 0.03
time_idx_step = np.arange(s.signal.time().shape[0])[s.signal.time() > strum_interval][0]
chord = np.zeros(arrays[0].shape[0] + 6 * time_idx_step)
i = 0
for arr in arrays:
    chord[i:i + arr.shape[0]] += arr
    i += time_idx_step

chord *= 0.95 / np.max(np.abs(chord))
plt.plot(chord)
du.listen_sig_array(chord)
