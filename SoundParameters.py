from recordtype import recordtype


class param(object):
    """
    A Class for storing the individual sound parameters
    """

    def __init__(self, name, value, info):
        self.name = name
        self.value = value
        self.info = info

    def change(self, new_value):
        self.value = new_value

class SoundParametersClass(object):

    def __init__(self):
        pass

    def info(self):
        for field, val in zip(self._asdict(), self._asdict().values()):
            print(field, ':')
            for key, entry in zip(val._asdict(), val._asdict().values()):
                print('\t', entry.name, ':', entry.value)

    def more_info(self):
        for field, val in zip(self._asdict(), self._asdict().values()):
            print(field, ':')
            for key, entry in zip(val._asdict(), val._asdict().values()):
                print('\t', entry.name, ':', entry.value, entry.string)


"""
Defining the default values and their descriptions for the sound parameters
"""

# General
octave_fraction = param('octave_fraction', 3, 'Fraction of the octave used for octave computations')
fft_range = param('fft_range', 2000, 'Maximum frequency in the Fourier transform plot')
onset_delay = param('onset_delay', 100, 'Delay before the onset (attack) to keep when trimming the signal')

# Envelop
frame_size = param('frame_size', 524, 'Number of samples in the array used to compute a point of the envelop')
hop_length = param('hop_length', 200, 'Number of samples between envelop points')

# Log Envelop
start_time = param('start_time', 0.01, 'First point in the log scale envelop')
min_window = param('min_window', None, 'Minimum window size for the log envelop computed from start_time by default')
max_window = param('max_window', 2048, 'Maximum window size for the log envelop in samples')

# Fundamental
min_freq = param('min_freq', 60, 'Minimum frequency for the fundamental finding algorithm (Hz)')
max_freq = param('max_freq', 2000, 'Maximum frequency for the fundamental finding algorithm (Hz)')
frame_length = param('frame_length', 1024, 'Frame length in samples to compute the fundamentals across the signal')

# Frequency bins to divide the signal
bass = param('bass', 100, 'Higher cutoff value for the bass bin (Hz), the lower value is zero')
mid = param('mid', 700, 'Higher cutoff value for the mid bin (Hz)')
highmid = param('highmid', 2000, 'Higher cutoff value for the highmid bin (Hz)')
uppermid = param('uppermid', 4000, 'Higher cutoff value for the uppermid bin (Hz)')
presence = param('presence', 6000, 'Higher cutoff value for the presence bin (Hz),'
                                   ' the brillance bin is above this frequency')

"""
Defining the sub parameter groups
"""

general = recordtype('general', [(octave_fraction.name, octave_fraction),
                                 (fft_range.name, fft_range),
                                 (onset_delay.name, onset_delay)])

envelop = recordtype('envelop', [(frame_size.name, frame_size),
                                 (hop_length.name, hop_length)])

log_envelop = recordtype('log_envelop', [(start_time.name, start_time),
                                         (min_window.name, min_window),
                                         (max_window.name, max_window)])

fundamental = recordtype('fundamental', [(min_freq.name, min_freq),
                                         (max_freq.name, max_freq),
                                         (frame_length.name, frame_length)])

bins = recordtype('bins', [(bass.name, bass),
                           (mid.name, mid),
                           (highmid.name, highmid),
                           (uppermid.name, uppermid),
                           (presence.name, presence)])

"""
Defining the global sound parameters structure.
"""

SoundParametersFunction = recordtype('SoundParams', [('general', general()),
                                                     ('bins', bins()),
                                                     ('envelop', envelop()),
                                                     ('log_envelop', log_envelop()),
                                                     ('fundamental', fundamental())])
