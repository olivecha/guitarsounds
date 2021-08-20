from IPython.display import display, clear_output
from guitarsounds import Signal, Sound, SoundPack
import ipywidgets as widgets
import io
import wave
import struct
import numpy as np

class guitarGUI(object):

    # Number of sound choice buttons
    button1 = widgets.Button(description="Single Sound")
    button2 = widgets.Button(description="Dual Sounds")
    button3 = widgets.Button(description="Multiple Sounds")

    # Ok Button to finish the sound import
    ok_button = widgets.Button(description="OK")

    # Done Button to finish the name choosing
    done_button = widgets.Button(description="Done")

    # ok Button to navigate the analysis choosing
    ok2_button = widgets.Button(description="OK")

    # go Button to do the analysis
    go_button = widgets.Button(description='Go')

    # loading bar
    style = {'bar_color': 'maroon', 'description_width': '100px'}
    load_bar = widgets.IntProgress(value=0, min=0, max=10, description='Importing files :', style=style)

    # Button Layout
    box_layout = widgets.Layout(justify_content='center', flex_flow='line', width='50%')
    items = [button1, button2, button3, ok_button]
    button_box = widgets.Box(children=items, layout=box_layout)

    # file selectors
    single_file_selector = widgets.FileUpload(accept='.wav', multiple=False)
    dual_file_selector_1 = widgets.FileUpload(accept='.wav', multiple=False)
    dual_file_selector_2 = widgets.FileUpload(accept='.wav', multiple=False)
    mult_file_selector = widgets.FileUpload(accept='.wav', multiple=True)

    # analysis drop downs
    # Single analysis drop down
    options = [('', 1),
               ('plot freq bins', Sound.plot_freq_bins),
               ('plot signal', (Signal.plot, 'signal')),
               ('plot envelop', (Signal.plot, 'envelop')),
               ('plot log envelop', (Signal.plot, 'log envelop')),
               ('plot FFT', (Signal.plot, 'fft')),
               ('plot FFT histogram', (Signal.plot, 'fft hist')),
               ('plot peaks', (Signal.plot, 'peaks')),
               ('plot peak damping', (Signal.plot, 'peak damping')),
               ('plot time damping', (Signal.plot, 'time damping'))]
    style = {'description_width': '150px'}
    single_drop_down = widgets.Dropdown(options=options, value=1, style=style,
                                        description='Choose an analysis : ')
    single_drop_down.rank = 'first'

    # Dual analysis drop down
    options = [('', 1),
               ('Compare Peaks', SoundPack.compare_peaks),
               ('Mirror FFT', SoundPack.fft_mirror),
               ('FFT difference', SoundPack.fft_diff)]
    style = {'description_width': '150px'}
    dual_drop_down = widgets.Dropdown(options=options, value=1, style=style,
                                      description='Choose an analysis : ')
    dual_drop_down.rank = 'first'

    # Multiple analysis drop down
    options = [('', 1),
               ('Stacked plot', SoundPack.plot),
               ('Compared plot', SoundPack.compare_plot),
               ('Frequency Bin plot', SoundPack.freq_bin_plot),
               ('Combine Envelops', SoundPack.combine_envelop),
               ('Print Fundamentals', SoundPack.fundamentals)]
    style = {'description_width': '150px'}
    mult_drop_down = widgets.Dropdown(options=options, value=1, style=style,
                                      description='Choose an analysis : ')
    mult_drop_down.rank = 'first'

    # Frequency bin choice drop down
    options = ['all', 'bass', 'mid', 'highmid', 'uppermid', 'presence', 'brillance']
    style = {'description_width': '150px'}
    bin_drop_down = widgets.Dropdown(options=options, value='all', style=style,
                                     description='Choose a frequency bin: ')
    bin_drop_down.rank = 'second'
    bin_drop_down.name = 'bin'

    # Plot type choice drop down
    options = [('Signal', 'signal'),
               ('Envelop', 'envelop'),
               ('Log Scale Envelop', 'log envelop'),
               ('Fourier Transform', 'fft'),
               ('Fourier Transform Histogram', 'fft hist'),
               ('Fourier Transform Peaks', 'peaks'),
               ('Peak Damping', 'peak damping'),
               ('Time Damping', 'time damping')]
    style = {'description_width': '150px'}
    plot_drop_down = widgets.Dropdown(options=options, value='signal', style=style,
                                      description='Choose a frequency bin: ')
    plot_drop_down.rank = 'second'
    plot_drop_down.name = 'plot'

    def __init__(self):

        # define the output
        self.output = widgets.Output()

        # Save analysis type
        self.analysis = None

        # display the buttons
        display(self.button_box, self.output)

        # Listen for clicks
        self.button1.on_click(self.on_single_button_clicked)
        self.button2.on_click(self.on_dual_button_clicked)
        self.button3.on_click(self.on_multiple_button_clicked)
        self.ok_button.on_click(self.on_1st_ok_button_clicked)
        self.go_button.on_click(self.on_go_button_clicked)

        # Drop downs associated to the analysis
        self.first_level_drop_down = {'Single': self.single_drop_down,
                                      'Dual': self.dual_drop_down,
                                      'Multiple': self.mult_drop_down}

    """
    Button click methods
    """

    def on_single_button_clicked(self, b):
        with self.output:
            display(self.single_file_selector)
        self.analysis = 'Single'
        self.state = 'file entry'

    def on_dual_button_clicked(self, b):
        with self.output:
            display(self.dual_file_selector_1)
            display(self.dual_file_selector_2)
        self.analysis = 'Dual'
        self.state = 'file entry'

    def on_multiple_button_clicked(self, b):
        with self.output:
            display(self.mult_file_selector)
        self.analysis = 'Multiple'
        self.state = 'file entry'

    def on_1st_ok_button_clicked(self, b):
        self.button_box.close()

        if self.analysis == 'Single':
            self.single_file_selector.close()
        elif self.analysis == 'Dual':
            self.dual_file_selector_1.close()
            self.dual_file_selector_2.close()
        elif self.analysis == 'Multiple':
            self.mult_file_selector.close()

        self.define_sound_names()

    def on_done_button_clicked(self, b):
        # remove text fields
        for text in self.sound_name_inputs:
            text.close()

        # remove the done button
        self.button_box.close()

        with self.output:
            display(self.load_bar)
        self.load_bar.observe(self.on_loaded_bar, names="value")

        self.load_bar.value += 1
        self.import_sound_files()
        self.load_bar.value = 10

    def on_2nd_ok_button_clicked(self, b):
        drop_down_value = self.current_drop_down.value

        if self.current_drop_down.rank == 'first':
            if self.analysis == 'Single':
                if drop_down_value == Sound.plot_freq_bins:
                    self.analysis_tuple = [drop_down_value]
                    self.current_drop_down.close()
                    self.current_drop_down = self.bin_drop_down
                    display(self.current_drop_down)

                elif type(drop_down_value) == tuple:
                    # variable and arg stored in a list
                    self.analysis_tuple = [*drop_down_value]
                    display(self.go_button)

            elif self.analysis == 'Dual':
                self.analysis_tuple = [drop_down_value]

            elif self.analysis == 'Multiple':
                if (drop_down_value == SoundPack.plot) or (drop_down_value == SoundPack.compare_plot):
                    self.analysis_tuple = [drop_down_value]
                    self.current_drop_down.close()
                    self.current_drop_down = self.plot_drop_down
                    display(self.current_drop_down)

                elif drop_down_value == SoundPack.freq_bin_plot:
                    self.analysis_tuple = [drop_down_value]
                    self.current_drop_down.close()
                    self.current_drop_down = self.bin_drop_down
                    display(self.current_drop_down)

                else:
                    self.analysis_tuple = [drop_down_value]
                    display(self.go_button)
                    self.go_button.on_click(self.on_go_button_clicked)

        if self.current_drop_down.rank == 'second':
            self.analysis_tuple.append(self.current_drop_down.value)
            display(self.go_button)
            self.go_button.on_click(self.on_go_button_clicked)

    def on_go_button_clicked(self, b):

        if self.analysis in ['Dual', 'Multiple']:

            # Call with no arguments
            if len(self.analysis_tuple) == 1:
                self.analysis_tuple[0](self.Pack)

            # Call with arguments
            elif len(self.analysis_tuple) == 2:
                self.analysis_tuple[0](self.Pack, self.analysis_tuple[1])

        elif self.analysis == 'Single':
            pass


    """
    Observe methods
    """

    def on_loaded_bar(self, change):
        # When the bar reaches the end
        if change["new"] == 10:
            # remove the bar
            self.load_bar.close()

            # change the button box and display with empty output
            self.button_box = widgets.Box(children=[self.ok2_button], layout=self.box_layout)
            self.output = widgets.Output()
            display(self.button_box, self.output)

            # display the drop down associated to the analysis
            self.current_drop_down = self.first_level_drop_down[self.analysis]
            with self.output:
                display(self.current_drop_down)
            self.ok2_button.on_click(self.on_2nd_ok_button_clicked)

    """
    Back end functions
    """

    def define_sound_names(self):

        self.button_box = widgets.Box(children=[self.done_button], layout=self.box_layout)
        self.output = widgets.Output()
        display(self.button_box, self.output)

        if self.analysis == 'Single':
            with self.output:
                self.file_names = [ky for ky in self.single_file_selector.value.keys()]
                sound_name_input = widgets.Text(value='',
                                                placeholder='sound name',
                                                description=self.file_names[0])
                display(sound_name_input)
                self.sound_name_inputs = [sound_name_input]

        elif self.analysis == 'Dual':
            name1 = [ky for ky in self.dual_file_selector_1.value.keys()][0]
            name2 = [ky for ky in self.dual_file_selector_2.value.keys()][0]
            self.file_names = [name1, name2]
            self.sound_name_inputs = []
            for file in self.file_names:
                sound_name_input = widgets.Text(value='',
                                                placeholder='sound name',
                                                description=file)
                display(sound_name_input)
                self.sound_name_inputs.append(sound_name_input)

        elif self.analysis == 'Multiple':
            self.file_names = [ky for ky in self.mult_file_selector.value.keys()]
            self.sound_name_inputs = []
            for file in self.file_names:
                sound_name_input = widgets.Text(value='',
                                                placeholder='sound name',
                                                description=file)
                display(sound_name_input)
                self.sound_name_inputs.append(sound_name_input)

        self.done_button.on_click(self.on_done_button_clicked)

    def import_sound_files(self):
        # Case for single file
        if self.analysis == 'Single':
            file_values = self.single_file_selector.value[self.file_names[0]]
            bites = file_values['content']
            audio = wave.open(io.BytesIO(bites))
            sr = audio.getframerate()
            samples = []
            self.load_bar.value += 1
            for _ in range(audio.getnframes()):
                frame = audio.readframes(1)
                samples.append(struct.unpack("h", frame)[0])
            self.load_bar.value += 1
            signal = np.array(samples) / 32768
            Sound_Input = (signal, sr)
            self.load_bar.value += 2
            self.Sons = Sound(Sound_Input, name=self.sound_name_inputs[0].value).condition(return_self=True)
            self.load_bar.value += 2

        # Case for two files
        elif self.analysis == 'Dual':
            self.Sons = []
            file_dicts = [self.dual_file_selector_1.value, self.dual_file_selector_2.value]
            for name, dic, name_input in zip(self.file_names, file_dicts, self.sound_name_inputs):
                self.load_bar.value += 1
                file_values = dic[name]
                bites = file_values['content']
                audio = wave.open(io.BytesIO(bites))
                sr = audio.getframerate()
                samples = []
                for _ in range(audio.getnframes()):
                    frame = audio.readframes(1)
                    samples.append(struct.unpack("h", frame)[0])
                signal = np.array(samples) / 32768
                Sound_Input = (signal, sr)
                self.Sons.append(Sound(Sound_Input, name=name_input.value).condition(return_self=True))
                self.load_bar.value += 2
            self.Pack = SoundPack(self.Sons, names=True)
            self.load_bar.value += 1

        # Case for multiple files
        elif self.analysis == 'Multiple':
            self.Sons = []
            for name, name_input in zip(self.file_names, self.sound_name_inputs):
                self.load_bar.value += 1
                file_values = self.mult_file_selector.value[name]
                bites = file_values['content']
                audio = wave.open(io.BytesIO(bites))
                sr = audio.getframerate()
                samples = []
                for _ in range(audio.getnframes()):
                    frame = audio.readframes(1)
                    samples.append(struct.unpack("h", frame)[0])
                signal = np.array(samples) / 32768
                Sound_Input = (signal, sr)
                self.Sons.append(Sound(Sound_Input, name=name_input.value).condition(return_self=True, verbose=False))
                self.load_bar.value += 1
            self.Pack = SoundPack(self.Sons, names=True)
            self.load_bar.value += 2

    def getpack(self):
        if hasattr(self, 'Pack'):
            return self.Pack

    def getsons(self):
        if hasattr(self, 'Sons'):
            return self.Sons
