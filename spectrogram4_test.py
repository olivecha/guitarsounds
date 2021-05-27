# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import numpy as np
from scipy.io.wavfile import read
import matplotlib.pyplot as plot
from matplotlib.pyplot import *
from scipy import fft
from scipy import signal
import matplotlib.pyplot as plt
import scipy
import math
import soundfile as sf

paths = []
samples = list(range(1, 9))
tries = list(range(1, 10))
for trie, sample in zip(tries, samples):
    paths.append('/test_leste/' + str(sample) + '-' + str(trie) + '.wav')

s1maxavg = (np.amax(x11) + np.amax(x12) + np.amax(x13) + np.amax(x14) + np.amax(x15) + np.amax(x16) + np.amax(
    x17) + np.amax(x18) + np.amax(x19)) / 9
s2maxavg = (np.amax(x21) + np.amax(x22) + np.amax(x23) + np.amax(x24) + np.amax(x25) + np.amax(x26) + np.amax(
    x27) + np.amax(x28) + np.amax(x29)) / 9
s3maxavg = (np.amax(x31) + np.amax(x32) + np.amax(x33) + np.amax(x34) + np.amax(x35) + np.amax(x36) + np.amax(
    x37) + np.amax(x38) + np.amax(x39)) / 9
s4maxavg = (np.amax(x41) + np.amax(x42) + np.amax(x43) + np.amax(x44) + np.amax(x45) + np.amax(x46) + np.amax(
    x47) + np.amax(x48) + np.amax(x49)) / 9
s5maxavg = (np.amax(x51) + np.amax(x52) + np.amax(x53) + np.amax(x54) + np.amax(x55) + np.amax(x56) + np.amax(
    x57) + np.amax(x58) + np.amax(x59)) / 9
s6maxavg = (np.amax(x61) + np.amax(x62) + np.amax(x63) + np.amax(x64) + np.amax(x65) + np.amax(x66) + np.amax(
    x67) + np.amax(x68) + np.amax(x69)) / 9
s7maxavg = (np.amax(x71) + np.amax(x72) + np.amax(x73) + np.amax(x74) + np.amax(x75) + np.amax(x76) + np.amax(
    x77) + np.amax(x78) + np.amax(x79)) / 9
s8maxavg = (np.amax(x81) + np.amax(x82) + np.amax(x83) + np.amax(x84) + np.amax(x85) + np.amax(x86) + np.amax(
    x87) + np.amax(x88) + np.amax(x89)) / 9

time_intervals = 0.04  # seconds

time = len(x1) / Fs
tranche = int(time / time_intervals)  # arrondi à l'integer le plus pret
xs1 = np.array_split(x1, tranche)
time = len(x2) / Fs
tranche = int(time / time_intervals)  # arrondi à l'integer le plus pret
xs2 = np.array_split(x2, tranche)
time = len(x3) / Fs
tranche = int(time / time_intervals)  # arrondi à l'integer le plus pret
xs3 = np.array_split(x3, tranche)
time = len(x4) / Fs
tranche = int(time / time_intervals)  # arrondi à l'integer le plus pret
xs4 = np.array_split(x4, tranche)
time = len(x5) / Fs
tranche = int(time / time_intervals)  # arrondi à l'integer le plus pret
xs5 = np.array_split(x5, tranche)
time = len(x6) / Fs
tranche = int(time / time_intervals)  # arrondi à l'integer le plus pret
xs6 = np.array_split(x6, tranche)
time = len(x7) / Fs
tranche = int(time / time_intervals)  # arrondi à l'integer le plus pret
xs7 = np.array_split(x7, tranche)
time = len(x8) / Fs
tranche = int(time / time_intervals)  # arrondi à l'integer le plus pret
xs8 = np.array_split(x8, tranche)
time = len(x9) / Fs
tranche = int(time / time_intervals)  # arrondi à l'integer le plus pret
xs9 = np.array_split(x9, tranche)

Y2 = [0]
for i in range(1, tranche):
    # define time
    t1 = np.arange(xs1[i].size) / float(Fs)
    # FFT of this
    n1 = xs1[i].size  # length of the signal
    T1 = n1 / float(Fs)
    frq1 = np.fft.fftfreq(n1, 1 / float(Fs))
    Y1 = fft(xs1[i]) / n1  # fft computing and normalization
    Y2 = Y2[0] + Y1

# plotting the data
t = np.arange(x1.size) / float(Fs)
subplot(3, 1, 1)
plot(t, x1, 'r')
xlabel('Time (seconds)')
ylabel('Amplitude')
grid()

# plotting the spectrum
subplot(3, 1, 2)
plot, semilogx(frq1[50:20000], abs(Y1[50:20000]), 'k')
xlabel('Freq (Hz)')
ylabel('|Y(freq)|')
grid()

frqparray = np.array(frqp)
subplot(3, 1, 3)
plot, semilogx(frq1[50:20000], abs(Y2[50:20000]), 'k')
xlabel('Freq (Hz)')
ylabel('|Y(freq)|')
grid()
