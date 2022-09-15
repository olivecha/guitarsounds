
class Parameter(object):
    """
    A class to store the individual sound parameters
    """
    def __init__(self, name, value, info):
        self.name = name
        self.value = value
        self.info = info


class ParameterSet(object):
    """
    A class to store multiple parameters as a set
    """
    def __init__(self, name, *parameters):
        self.name = name
        for parameter in parameters:
            setattr(self, parameter.name, parameter)

    def info(self):
        print(self.name)
        parameters = [parameter for parameter in self.__dict__.values() if type(parameter) != str]
        for parameter in parameters:
            print('\t', parameter.name, ':', parameter.value)


class GlobalParameters(object):
    """
    A class to store the parameter sets and to be used to assign parameter values to the different functions
    """

    def __init__(self, *parameter_sets):
        for parameter_set in parameter_sets:
            setattr(self, parameter_set.name, parameter_set)

    def info(self):
        """
        Prints the name and values of every parameter
        """
        for parameter_set in self.__dict__.values():
            print(parameter_set.name)
            parameters = [parameter for parameter in parameter_set.__dict__.values() if type(parameter) != str]
            for parameter in parameters:
                print('\t', parameter.name, ':', parameter.value)

    def more_info(self):
        """
        Prints the name, value and info string of every parameter
        """
        for parameter_set in self.__dict__.values():
            print(parameter_set.name)
            parameters = [parameter for parameter in parameter_set.__dict__.values() if type(parameter) != str]
            for parameter in parameters:
                info = '[' + parameter.info + ']'
                print('\t', parameter.name, ':', parameter.value, info)

    def change(self, name, value):
        """
        Change the parameter with the name `name` from its current value to the specified value
        """
        for parameter_set in self.__dict__.values():
            parameters = [parameter for parameter in parameter_set.__dict__.values() if type(parameter) != str]
            for parameter in parameters:
                if parameter.name == name:
                    parameter.value = value


def sound_parameters():
    """
    Function returning the instance of the sound parameters
    """
    # General
    octave_fraction = Parameter('octave_fraction', 3, 'Fraction of the octave used for octave computations')
    fft_range = Parameter('fft_range', 2000, 'Maximum frequency in the Fourier transform plot')
    general = ParameterSet('general', octave_fraction, fft_range)

    # Onset
    onset_delay = Parameter('onset_delay', 100, 'Delay before the onset (attack) to keep when trimming the signal')
    onset_time = Parameter('onset_time', 0.005, 'Time interval used to detect the onset in seconds')
    onset = ParameterSet('onset', onset_delay, onset_time)

    # Envelop
    frame_size = Parameter('frame_size', 301, 'Number of samples in the array used to compute a point of the envelop')
    hop_length = Parameter('hop_length', 200, 'Number of samples between envelop points')
    envelop = ParameterSet('envelop', frame_size, hop_length)

    # Log Envelop
    start_time = Parameter('start_time', 0.01, 'First point in the log scale envelop')
    min_window = Parameter('min_window', None, 'Minimum window size for the log envelop computed from start_time by '
                                               'default')
    max_window = Parameter('max_window', 2048, 'Maximum window size for the log envelop in samples')
    log_envelop = ParameterSet('log_envelop', start_time, min_window, max_window)

    # Fundamental
    min_freq = Parameter('min_freq', 60, 'Minimum frequency for the fundamental finding algorithm (Hz)')
    max_freq = Parameter('max_freq', 2000, 'Maximum frequency for the fundamental finding algorithm (Hz)')
    frame_length = Parameter('frame_length', 1024, 'Frame length in samples to compute the fundamentals in the signal')
    fundamental = ParameterSet('fundamental', min_freq, max_freq, frame_length)

    # Frequency bins to divide the signal
    bass = Parameter('bass', 100, 'Higher cutoff value for the bass bin (Hz), the lower value is zero')
    mid = Parameter('mid', 700, 'Higher cutoff value for the mid bin (Hz)')
    highmid = Parameter('highmid', 2000, 'Higher cutoff value for the highmid bin (Hz)')
    uppermid = Parameter('uppermid', 4000, 'Higher cutoff value for the uppermid bin (Hz)')
    presence = Parameter('presence', 6000, 'Higher cutoff value for the presence bin (Hz),'
                                           ' the brilliance bin is above this frequency')
    bins = ParameterSet('bins', bass, mid, highmid, uppermid, presence)

    # Damping
    lower_threshold = Parameter('lower_threshold', 0.05, 'lower amplitude threshold for the time damping analysis')
    damping = ParameterSet('damping', lower_threshold)
    
    # Default signal trim times
    E = Parameter('E2', 4.0, 'Default trim time for the E2 note')
    A = Parameter('A2', 3.5, 'Default trim time for the A2 note') 
    D = Parameter('D3', 3.5, 'Default trim time for the D3 note')
    G = Parameter('G3', 3.0, 'Default trim time for the G3 note') 
    B = Parameter('B3', 3.0, 'Default trim time for the B3 note')
    e = Parameter('E4', 2.5, 'Default trim time for the E4 note')
    trim_times = ParameterSet('trim', E, A, D, G, B, e)

    SoundParameters = GlobalParameters(general, onset, envelop, log_envelop, fundamental, bins, damping, trim_times)

    return SoundParameters
