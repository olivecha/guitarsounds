import guitarsounds
from guitarsounds import Sound
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# -----Module variables----- #
MIN_A_VALUE = 1.216110966595552e-21 
MAX_A_VALUE = 2.956038122585772e-11 
MEAN_A_VALUE = (MIN_A_VALUE + MAX_A_VALUE) / 2 
MIN_B_VALUE = 241.88870914170292 
MAX_B_VALUE = 476.8504184761331 
MEAN_B_VALUE = (MIN_B_VALUE + MAX_B_VALUE) / 2


def exp_from_ab(a, b):
    """ Create an exponential callable from a and b parameters """
    def custom_exp(t):
        return a * np.exp(t * b)
    return custom_exp
    

def get_notes_names():
    """ get the note names of the example sounds""" 
    root = 'example_sounds/Wood_Guitar/'
    files = np.sort([root + file for file in os.listdir(root)])
    notes = [file[-6:-4] for file in files]
    return notes


def load_nylon_sounds(as_dict=False):
    """ Load the example wood sounds"""
    nylon_root = 'example_sounds/Nylon_Guitar/'
    nylon_files = np.sort([nylon_root + file for file in os.listdir(nylon_root)])
    nylon_sounds = []
    for nf in nylon_files:
        s = Sound(str(nf))
        s = s.condition(return_self=True, auto_trim=True)
        nylon_sounds.append(s)
    if as_dict:
        nylon_dict = {}
        notes = get_notes_names()
        for i, n in enumerate(notes):
            nylon_dict[n] = nylon_sounds[i]
        return nylon_dict
    else:
        return nylon_sounds


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
    idx = np.arange(t.shape[0])[t > time][0]
    return idx


def frequency_index(signal, freq):
    """ Return the approximate index of the frequency in the signal fft"""
    f = signal.fft_frequencies()
    idx = np.arange(f.shape[0])[f > freq][0]
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


def get_expenv(p):
    """ 
    Generate an exponential onset based on curves fitted on real signals
    :param p: a float, if p is between -1 and 1 the fitted envelop will be 
    within the experimental range
    for p > 1 or p < -1 the shape of the onset envelop is extrapolated.
    """
    # Hard coded bounds on envelop onset parameters
    a_min = 1.216110966595552e-21
    a_max = 2.956038122585772e-11
    b_min = 241.88870914170292
    b_max = 476.8504184761331
    
    if p > 1:
        p = np.sqrt(p)
        # a_max is the highest curve
        a = a_min + (p / 2 + 0.5) * (a_max - a_min)
        # b_min is the highest curve
        b = b_min + (1 - (p / 2 + 0.5)) * (b_max - b_min)
        
    elif p < -1:
        b = b_min + (1 - (p / 2 + 0.5)) * (b_max - b_min)
        p = 1 / np.abs(p)
        a = a_min + (p / 2 + 0.5) * (a_max - a_min)
        
    else:
        # a_max is the highest curve
        a = a_min + (p / 2 + 0.5) * (a_max - a_min)
        # b_min is the highest curve
        b = b_min + (1 - (p / 2 + 0.5)) * (b_max - b_min)
            
    # Correct the value at t = 0.1 at run time
    t1 = np.linspace(0, 0.6, 1000)
    env_exp_draft = a * np.exp(t1 * b)
    itrp = interp1d(env_exp_draft, t1)
    offset = itrp(1) - 0.1
    
    # Create the callable for the current envelop value
    def expenv(t):
        return a * np.exp((t + offset) * b)
    
    return expenv


def get_low_exp(start, stop, step):
    """ 
    Return the lower experimental curve associated to the envelop
    associated to the experimental sound data
    """
    a_min = 1.216110966595552e-21
    b_max = 476.8504184761331
    t_interp = np.linspace(start, stop * 2, step * 2)
    exp_low_draft = a_min * np.exp(t_interp * b_max)
    low_interp = interp1d(exp_low_draft, t_interp)
    off_set = low_interp(1) - 0.1
    t = np.linspace(start, stop, step)
    exp_low = a_min * np.exp((t + off_set) * b_max)    
    return exp_low


def get_hi_exp(start, stop, step):
    """
    Return the lower experimental curve associated to the envelop
    associated to the experimental sound data
    """
    a_max = 2.956038122585772e-11
    b_min = 241.88870914170292
    t_interp = np.linspace(start, stop * 2, step * 2)
    exp_hi_draft = a_max * np.exp(t_interp * b_min)
    hi_interp = interp1d(exp_hi_draft, t_interp)
    off_set = hi_interp(1) - 0.1
    t = np.linspace(start, stop, step)
    exp_hi = a_max * np.exp((t + off_set) * b_min)
    return exp_hi


