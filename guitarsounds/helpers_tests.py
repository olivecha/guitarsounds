from guitarsounds import Signal
from guitarsounds.parameters import sound_parameters
from guitarsounds.utils import load_wav as load
from guitarsounds.utils import resample

import os
from random import randint


def get_rnd_audio_file():
    """ Get a random soundfile for the included example sounds"""
    wood_root = os.path.join('..', 'example_sounds', 'Wood_Guitar')
    carbon_root = os.path.join('..', 'example_sounds', 'Carbon_Guitar')
    wood_files = [os.path.join(wood_root, file) for file in os.listdir(wood_root)]
    carbon_files = [os.path.join(carbon_root, file) for file in os.listdir(carbon_root)]
    all_files = wood_files + carbon_files
    idx = randint(0, len(all_files) - 1)
    return all_files[idx]


def get_rnd_test_Signal():
    """Get a random guitarsounds.Signal instance from the example sounds"""
    random_signal_file = get_rnd_audio_file()
    signal_data, sample_rate = load(random_signal_file)
    signal_data = resample(signal_data, sample_rate, 22050)
    signal = Signal(signal_data, 22050, sound_parameters())
    return signal


def get_ref_test_Signal():
    reference_signal_file = os.path.join('..', 'example_sounds', 'Carbon_Guitar', 'Carbon_G3.wav')
    signal_data, sample_rate = load(reference_signal_file)
    signal_data = resample(signal_data, sample_rate, 22050)
    signal = Signal(signal_data, 22050, sound_parameters())
    return signal
