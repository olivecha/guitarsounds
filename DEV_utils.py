import guitarsounds
from guitarsounds import Sound
import sys, os
import inspect
import numpy as np
import matplotlib.pyplot as plt

def get_notes_names():
    """ get the note names of the example sounds"""
    root = 'example_sounds/Wood_Guitar/'
    files = np.sort([root + file for file in os.listdir(root)])
    notes = [file[-6:-4] for file in files]
    return notes

def load_wood_sounds(as_dict=False):
    """ Load the example wood sounds"""
    wood_root = 'example_sounds/Wood_Guitar/'
    wood_files = np.sort([wood_root + file for file in os.listdir(wood_root)])
    wood_sounds = []
    for wf in wood_files:
        s = Sound(str(wf))
        s = s.condition(return_self=True, auto_trim=True)
        wood_sounds.append(s)
    if as_dict:
        wood_dict = {}
        notes = get_notes_names()
        for i, n in enumerate(notes):
            wood_dict[n] = wood_sounds[i]
        return wood_dict
    else:
        return wood_sounds

def load_carbon_sounds(as_dict=False):
    """ Load the example carbon guitar sounds"""
    carbon_root = 'example_sounds/Carbon_Guitar/'
    carbon_files = np.sort([carbon_root + file for file in os.listdir(carbon_root)])
    carbon_sounds = []
    for cf in carbon_files:
        s = Sound(str(cf))
        s = s.condition(return_self=True, auto_trim=True)
        carbon_sounds.append(s)  
    if as_dict:
        carbon_dict = {}
        notes = get_notes_names()
        for i, n in enumerate(notes):
            carbon_dict[n] = carbon_sounds[i]
        return carbon_dict
    else:
        return carbon_sounds
    
def time_index(signal, time):
    """ Return the approximate index of time in signal"""
    t = signal.time()
    idx = np.arange(t.shape[0])[t>time][0]
    return idx

def frequency_index(signal, freq):
    """ Return the approximate index of the frequency in the signal fft"""
    f = signal.fft_frequencies()
    idx = np.arange(f.shape[0])[f>freq][0]
    return idx

def listen_sig_array(arr):
    """ listen a signal array """
    arr2sig(arr).listen()
    
def arr2sig(arr):
    """ Convert a signal array to a Signal class instance"""
    return guitarsounds.Signal(arr, 22050, guitarsounds.parameters.sound_parameters())

def scomp(arr, s):
    """ Compare the sound of a signal array with an existing sound"""
    ns = arr2sig(arr)
    print('Sound')
    s.signal.trim_time(ns.time()[-1]).normalize().listen()
    print('Array')
    listen_sig_array(arr)
    
def bigfig():
    """ Make the current figure big"""
    plt.gcf().set_size_inches(10, 6)

def arr2notedict(soundlist):
    """ Converts a list of sounds to a dict with notes as keys """
    notes = get_notes_names()
    note_dict = {}
    for n, s in zip(notes, soundlist):
        note_dict[n] = s
    return note_dict

def get_onset_env():
    pass        
