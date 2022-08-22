import resampy
import wave
import struct
import scipy.signal

wood_sounds = du.load_wood_sounds()

print('__ guitarsound loaded sound __')
ref_sig = wood_sounds['D4'].signal
print(f'Sample rate : {ref_sig.sr}')
print(f'No of data points : {ref_sig.signal.shape[0]}')
ref_sig.listen()


print('\n __ wave loaded sound __')
file = 'example_sounds/Wood_Guitar/Wood_D4.wav'
audio = wave.open(file)
sr = audio.getframerate()
samples = []

for _ in range(audio.getnframes()):
    frame = audio.readframes(1)
    samples.append(struct.unpack("h", frame)[0])

signal = np.array(samples) / 32768
sig = du.arr2sig(signal, sr)
sig = sig.trim_onset()
print(f'Sample rate : {sig.sr}')
print(f'No of data points : {sig.signal.shape[0]}')
sig = sig.trim_time(ref_sig.time()[-1])
sig.listen()

print('\n __ scipy resampled sound __')
sig_data = scipy.signal.resample(sig.signal, num=int(ref_sig.time()[-1]*ref_sig.sr))
sci_sig = du.arr2sig(sig_data, ref_sig.sr)
print(f'Sample rate : {sci_sig.sr}')
print(f'No of data points : {sci_sig.signal.shape[0]}')
sci_sig.listen()

print('\n __ resampy resampled sound __')
#sig_data = resampy.resample(sig.signal, sr, ref_sig.sr)
rsy_sig = du.arr2sig(sig_data, ref_sig.sr)
print(f'Sample rate : {rsy_sig.sr}')
print(f'No of data points : {rsy_sig.signal.shape[0]}')
rsy_sig.listen()
