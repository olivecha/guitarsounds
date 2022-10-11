import guitarsounds
from guitarsounds import Sound
import sys, os
import inspect
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import librosa
import librosa.display
import DEV_utils as du
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d, interp2d, InterpolatedUnivariateSpline, RectBivariateSpline
import scipy.signal

wood_sounds = du.load_wood_sounds()
wood_sounds = du.arr2notedict(wood_sounds)
notes = du.get_notes_names()


# FUNCTIONS
def noabsfft(sig):
    """
    Real valued fft (not norm)
    """
    # Double the signal for better precision
    fft_sig = np.hstack([sig.signal, np.flip(sig.signal)])
    fft = np.fft.fft(fft_sig)
    fft = np.real(fft)
    fft = fft[:fft.size//2]
    return fft

def fft2sig(fft):
    """
    Transform a real truncated FFT to a signal (ifft)
    """
    fft = np.hstack([fft, np.flip(fft)])
    ifft_sig = np.fft.ifft(fft)
    ifft_sig = np.real(ifft_sig)
    ifft_sig = ifft_sig[:ifft_sig.size//2]
    return ifft_sig

def apply_onset_ifft(sigarr, expenv_param=0, sr=22050):
    """ Apply the onset envelop to a inverse FFT signal """
    env = du.get_expenv(expenv_param)
    # Apply it to time = 0.0 - 0.1 s
    time = ifft_time(sigarr, sr=sr)
    t_idx = np.arange(time.shape[0])[time < 0.1][-1]
    sigarr[:t_idx] = env(time[:t_idx]) * sigarr[:t_idx]
    return sigarr

def ifft_time(sigarr, sr=22050):
    """ Time vector of an inverse fourier transform """
    time = np.arange(0, sigarr.size/22050, 1/22050)
    return time

def fft_freq(fftarr, sr=22050):
    freq = np.fft.fftfreq(fftarr.size*2, 1/sr)
    return freq[:freq.size//2]
    
def fadeout_sigarr(sigarr, fade_len=20):
    """ Fade out a signal array """
    sigarr[-fade_len:] = sigarr[-fade_len:] * np.linspace(1, 0.1, fade_len)
    return sigarr

def gaussian(x, mu, sigma):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))


def gaussian_frequency_perturbation_ifft_soundgen():
    sig = wood_sounds['A5'].signal

    fft = noabsfft(sig)
    
    freq_vals = fft_freq(fft)
    idx_arr = np.logical_and(freq_vals>(220-5), freq_vals<(220+5))
    pert = 0.2*gaussian(freq_vals[idx_arr], 220, 2.5)
    fft_pert = fft.copy()
    fft_pert[idx_arr] = fft_pert[idx_arr]*pert
        
    ifft_sig1 = fft2sig(fft_pert)
    #ifft_sig1 = apply_onset_ifft(ifft_sig1)
    ifft_sig1 = fadeout_sigarr(ifft_sig1)
    
    ifft_sig2 = fft2sig(fft)
    #ifft_sig2 = apply_onset_ifft(ifft_sig2)
    ifft_sig2 = fadeout_sigarr(ifft_sig2)
    
    time = ifft_time(ifft_sig1)
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    
    ax = axs[0]
    plt.sca(ax)
    sig.plot.signal(label='Original')
    ax.plot(time, ifft_sig1, alpha=0.3, label='Generated')
    
    ax = axs[1]
    plt.sca(ax)
    ax.plot(fft_freq(fft), np.abs(fft), label='Original FFT')
    ax.plot(fft_freq(fft), np.abs(fft_pert), label='FFT with perturbation')
    ax.set_xlim(0, 1500)
    ax.set_yscale('symlog')
    
    print('Original : ')
    sig.listen()
    
    print('Ifft Generated (clean)')
    du.listen_sig_array(ifft_sig2, sr=22050)
    
    print('IFFT Generated (attenuated 220 Hz) : ')
    du.listen_sig_array(ifft_sig1, sr=22050)
        
        
def ramp_up_ifft_soundgen_perturbation():
    # Ramp up perturbation
    sig = wood_sounds['A5'].signal
    
    fft = noabsfft(sig)
    
    pert = np.linspace(1, 10, fft.size)
    fft_pert = fft.copy()
    fft_pert = fft_pert*pert
        
    ifft_sig1 = fft2sig(fft_pert)
    ifft_sig1 = apply_onset_ifft(ifft_sig1)
    ifft_sig1 = fadeout_sigarr(ifft_sig1)
    
    ifft_sig2 = fft2sig(fft)
    ifft_sig2 = apply_onset_ifft(ifft_sig2)
    ifft_sig2 = fadeout_sigarr(ifft_sig2)
    
    time = ifft_time(ifft_sig1)
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    
    ax = axs[0]
    plt.sca(ax)
    plt.plot(fft_freq(fft), (pert-1)/10)
    ax.plot(fft_freq(fft), np.abs(fft)/np.max(np.abs(fft)))
    ax.set_xlabel('Frequence')
    ax.set_ylabel('Perturbation')
    
    ax = axs[1]
    plt.sca(ax)
    ax.plot(fft_freq(fft), np.abs(fft_pert), color='red', label='FFT with perturbation')
    ax.plot(fft_freq(fft), np.abs(fft), color='blue', label='Original FFT')
    ax.set_yscale('symlog')
    ax.legend()
    
    print('Original : ')
    sig.listen()
    
    print('Ifft Generated (clean)')
    du.listen_sig_array(ifft_sig2, sr=22050)
    
    print('IFFT Generated (Incresed higher frequencies) : ')
    du.listen_sig_array(ifft_sig1, sr=22050)
    
    
def ramp_down_ifft_soundgen_perturbation():
    # Ramp up perturbation
    sig = wood_sounds['A5'].signal
    
    fft = noabsfft(sig)
    
    pert = np.exp(-np.linspace(0, 10, fft.size))
    fft_pert = fft.copy()
    fft_pert = fft_pert*pert
    fft_pert = 1.1*np.max(np.abs(fft)) * fft_pert / np.max(np.abs(fft_pert))
        
    ifft_sig1 = fft2sig(fft_pert)
    ifft_sig1 = apply_onset_ifft(ifft_sig1)
    ifft_sig1 = fadeout_sigarr(ifft_sig1)
    
    ifft_sig2 = fft2sig(fft)
    ifft_sig2 = apply_onset_ifft(ifft_sig2)
    ifft_sig2 = fadeout_sigarr(ifft_sig2)
    
    time = ifft_time(ifft_sig1)
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    
    ax = axs[0]
    plt.sca(ax)
    plt.plot(fft_freq(fft), (pert))
    ax.plot(fft_freq(fft), np.abs(fft)/np.max(np.abs(fft)))
    ax.set_xlabel('Frequence')
    ax.set_ylabel('Perturbation')
    
    ax = axs[1]
    plt.sca(ax)
    ax.plot(fft_freq(fft), np.abs(fft), color='blue', label='Original FFT')
    ax.plot(fft_freq(fft), np.abs(fft_pert), color='red', label='FFT with perturbation')
    ax.set_yscale('symlog')
    ax.legend()
    
    print('Original : ')
    sig.listen()
    
    print('Ifft Generated (clean)')
    du.listen_sig_array(ifft_sig2, sr=22050)
    
    print('IFFT Generated (Incresed higher frequencies) : ')
    du.listen_sig_array(ifft_sig1, sr=22050)