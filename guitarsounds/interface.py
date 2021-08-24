from IPython.display import display, clear_output, HTML
from guitarsounds.analysis import Signal, Sound, SoundPack
import ipywidgets as widgets
import matplotlib.pyplot as plt
import io
import wave
import struct
import numpy as np

def generate_error_widget(text):
    return widgets.HTML('<p style="color:#CC4123;">' + text + '</p>')


class guitarGUI(object):
    # Output layout
    out_layout = {'border': '1px solid black'}

    # Box Layout
    box_layout = widgets.Layout(align_items='stretch', flex_flow='line', width='50%')

    # Attribute for output layout
    output = widgets.Output(layout={'border': '1px solid black'})

    # analysis drop downs
    # Single analysis drop down
    options = [('', 1),
               ('Listen Sound', Signal.listen),
               ('Listen frequency bins', Sound.listen_freq_bins),
               ('Frequency bin plot', Sound.plot_freq_bins),
               ('Signal plot', (Signal.plot, 'signal')),
               ('Envelop plot', (Signal.plot, 'envelop')),
               ('Log-envelop plot', (Signal.plot, 'log envelop')),
               ('Fourier transform plot', (Signal.plot, 'fft')),
               ('Fourier transform histogram', (Signal.plot, 'fft hist')),
               ('Peaks plot', (Signal.plot, 'peaks')),
               ('Peak damping plot', (Signal.plot, 'peak damping')),
               ('Time damping plot', (Signal.plot, 'time damping')),
               ('Timbre attributes plot', (Signal.plot, 'timbre')),
               ('Frequency damping values', Sound.peak_damping)]
    style = {'description_width': '150px'}
    single_drop_down = widgets.Dropdown(options=options, value=1, style=style,
                                        description='Choose an analysis : ')
    single_drop_down.rank = 'first'

    # Dual analysis drop down
    options = [('', 1),
               ('Compare Peaks', SoundPack.compare_peaks),
               ('Mirror FFT', SoundPack.fft_mirror),
               ('FFT difference', SoundPack.fft_diff),
               ('Stacked plot', SoundPack.plot),
               ('Compared plot', SoundPack.compare_plot),
               ('Frequency Bin plot', SoundPack.freq_bin_plot),
               ('Print Fundamentals', SoundPack.fundamentals),]
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
    options = [('', 1),
               ('all', 'all'),
               ('bass', 'bass'),
               ('mid', 'mid'),
               ('highmid', 'highmid'),
               ('uppermid', 'uppermid'),
               ('presence', 'presence'),
               ('brillance', 'brillance')]
    style = {'description_width': '150px'}
    bin_drop_down = widgets.Dropdown(options=options, value='all', style=style,
                                     description='Choose a frequency bin: ')
    bin_drop_down.rank = 'second'
    bin_drop_down.name = 'bin'

    # Plot type choice drop down
    options = [('', 1),
               ('Signal', 'signal'),
               ('Envelop', 'envelop'),
               ('Log Scale Envelop', 'log envelop'),
               ('Fourier Transform', 'fft'),
               ('Fourier Transform Histogram', 'fft hist'),
               ('Fourier Transform Peaks', 'peaks'),
               ('Peak Damping', 'peak damping'),
               ('Time Damping', 'time damping'),
               ('Timbre Attributes', 'timbre')]
    style = {'description_width': '150px'}
    plot_drop_down = widgets.Dropdown(options=options, value='signal', style=style,
                                      description='Choose a plot type: ')
    plot_drop_down.rank = 'second'
    plot_drop_down.name = 'plot'

    def __init__(self):
        """
        Here We display the three file choosing buttons matched with the
        three types of analyses when one is clicked the user is prompted
        to choose files.

        When files are chosen  the user press the 'Ok' Button and the
        Program advances to defining names see `.on_ok_button_clicked_1`.
        """

        # __ Buttons __
        # Number of sound choice buttons
        self.button1 = widgets.Button(description="Single Sound")
        self.button2 = widgets.Button(description="Dual Sounds")
        self.button3 = widgets.Button(description="Multiple Sounds")

        # Ok, Done and Go Buttons
        self.ok_button = widgets.Button(description="Ok")
        self.done_button = widgets.Button(description="Done")
        self.go_button = widgets.Button(description='Go')

        # Normalize toggle button
        self.toggle_normalize_button = widgets.Button(description='Normalize')
        # Associated attribute to normalize the Sounds for the method called
        self.normalize = False

        # Button box when the GUI starts
        self.button_box = widgets.Box(children=[self.button1,
                                                self.button2,
                                                self.button3,
                                                self.ok_button], layout=self.box_layout)

        # Load bar when importing sounds
        self.load_bar = widgets.IntProgress(value=5, min=0, max=10,
                                            description='Importing sound files :',
                                            style={'bar_color': '#6495ED',
                                                   'description_width': '140px'}, )

        # File selectors for uploading files into the program
        self.single_file_selector = widgets.FileUpload(accept='.wav', multiple=False)
        self.dual_file_selector_1 = widgets.FileUpload(accept='.wav', multiple=False)
        self.dual_file_selector_2 = widgets.FileUpload(accept='.wav', multiple=False)
        self.mult_file_selector = widgets.FileUpload(accept='.wav', multiple=True)

        # Dict with drop down methods to display the drop down associated to
        # the analysis
        self.first_level_drop_down = {'Single': self.single_drop_down,
                                      'Dual': self.dual_drop_down,
                                      'Multiple': self.mult_drop_down}

        # Save analysis type
        self.analysis = None

        # Define the current state of the program
        self.state = 'start'

        # Listen for clicks on the first button panel
        self.button1.on_click(self.on_single_button_clicked)
        self.button2.on_click(self.on_dual_button_clicked)
        self.button3.on_click(self.on_multiple_button_clicked)
        self.ok_button.on_click(self.on_ok_button_clicked_1)
        self.disable_file_selection(False)

        # display the buttons
        display(self.button_box)

    """
    File Choosing Interface Button Click Methods
    """

    def on_single_button_clicked(self, b):
        """
        Displays the single file selector, allowing the user to choose
        one file.
        """
        clear_output(wait=True)

        output = widgets.Output(layout={'border': '1px solid black'})
        self.disable_file_selection(True)
        with output:
            display(self.single_file_selector)

        self.analysis = 'Single'
        self.state = 'file entry'

        display(self.button_box)
        display(output)

    def on_dual_button_clicked(self, b):
        """
        Displays two single file selectors, allowing the user
        to choose two files.
        """
        clear_output(wait=True)

        output = widgets.Output(layout={'border': '1px solid black'})
        self.disable_file_selection(True)
        with output:
            display(self.dual_file_selector_1)
            display(self.dual_file_selector_2)

        self.analysis = 'Dual'
        self.state = 'file entry'

        display(self.button_box)
        display(output)

    def on_multiple_button_clicked(self, b):
        """
        Displays a multiple file selector allowing the user
        to select multiple files
        """
        clear_output(wait=True)

        output = widgets.Output(layout={'border': '1px solid black'})
        self.disable_file_selection(True)
        with output:
            display(self.mult_file_selector)

        self.analysis = 'Multiple'
        self.state = 'file entry'

        display(self.button_box)
        display(output)

    def disable_file_selection(self, state):
        self.button1.disabled = state
        self.button2.disabled = state
        self.button3.disabled = state

    def on_ok_button_clicked_1(self, b):
        """
        The user clicks this button when he is done choosing files
        """
        # Clear the output
        clear_output(wait=True)

        # Check if the user did good when choosing files
        file_selectors = [self.single_file_selector,
                          self.dual_file_selector_1,
                          self.dual_file_selector_2,
                          self.mult_file_selector]
        files_where_chosen = False
        for file_selector in file_selectors:
            if file_selector.value != {}:
                files_where_chosen = True

        # If the
        if files_where_chosen:
            self.define_sound_names()

        else:
            output = widgets.Output(layout={'border': '1px solid black'})
            with output:
                if self.analysis == 'Single':
                    display(self.single_file_selector)
                elif self.analysis == 'Dual':
                    display(self.dual_file_selector_1)
                    display(self.dual_file_selector_2)
                elif self.analysis == 'Multiple':
                    display(self.mult_file_selector)
                else:
                    error = generate_error_widget('Chose an analysis type')
                    display(error)
                if self.analysis in ['Single', 'Dual', 'Multiple']:
                    error = generate_error_widget('No sound was chosen')
                    display(error)

            display(self.button_box)
            display(output)

    def on_ok_button_clicked_2(self, b):
        """
        Method to make the "Ok" button interact with the
        analysis method choice.

        __ when interface.state = 'method choice' __
        - The "Ok" and "Go" buttons appears after the loading bar is done
        - The drop down corresponds to the methods associated to
        the analysis

        """
        # Clear the Output
        clear_output(wait=True)
        output = widgets.Output(layout=self.out_layout)

        # Save the drop down value
        drop_down_value = self.current_drop_down.value

        if self.state == 'method choice':
            # If we only analyse a single sound
            if self.analysis == 'Single':

                # Special case when the method is the frequency bin plot
                if drop_down_value == Sound.plot_freq_bins:
                    self.analysis_tuple = [drop_down_value]  # Store the method
                    # Change the drop down to frequency bin choice
                    self.current_drop_down = self.bin_drop_down
                    self.state = 'method choice 2'  # a second choice is needed
                    self.display = 'plot'

                elif drop_down_value in [Sound.peak_damping, Sound.listen_freq_bins, Signal.listen]:
                    self.analysis_tuple = [drop_down_value]  # store the method
                    self.state = 'display'  # ready to display
                    self.display = 'print'

                # all other methods are tuples (fun, arg) -> fun(arg)
                elif type(drop_down_value) == tuple:
                    # store method and arg in a list
                    self.analysis_tuple = [*drop_down_value]
                    self.state = 'display'  # ready to display
                    self.display = 'plot'

                elif drop_down_value == 1:
                    error = generate_error_widget('No analysis method was chosen')
                    with output:
                        display(error)

            elif self.analysis in ['Dual', 'Multiple']:  # Case when two sounds or multiple sounds are being analysed
                # Special case for the frequency bin plot
                if drop_down_value == SoundPack.freq_bin_plot:
                    self.analysis_tuple = [drop_down_value]  # Store the method
                    # Update the drop down to frequency bin choice
                    self.current_drop_down = self.bin_drop_down
                    self.state = 'method choice 2'  # a second choice is needed
                    self.display = 'plot'

                # Case for plot methods
                elif (drop_down_value == SoundPack.plot) or (drop_down_value == SoundPack.compare_plot):
                    self.analysis_tuple = [drop_down_value]  # Store the method
                    # Update the drop down to the plot drop down
                    self.current_drop_down = self.plot_drop_down
                    self.state = 'method choice 2'  # a second choice is needed
                    self.display = 'plot'

                elif drop_down_value == 1:
                    error = generate_error_widget('No analysis method was chosen')
                    with output:
                        display(error)

                # Case for methods with no arguments
                else:
                    if drop_down_value == SoundPack.fundamentals:
                        self.display = 'print'
                    else:
                        self.display = 'plot'

                    self.analysis_tuple = [drop_down_value]  # store the method
                    self.state = 'display'

        # Case when the method is chosen and an argument needs to be added 'method choice 2'
        elif self.state == 'method choice 2':

            # add the arg part to the analysis tuple
            self.analysis_tuple.append(self.current_drop_down.value)
            self.state = 'display'

        # Coming back from the display the state is redefined and we restart
        elif self.state == 'analysis displayed':
            self.state = 'method choice'

        # If the button is pressed and the method is defined, the go button is enabled
        if self.state == 'display':
            self.go_button.disabled = False
            self.ok_button.disabled = True

        # Actualize the button box and display
        children = [self.ok_button, self.go_button, self.toggle_normalize_button]
        self.button_box = widgets.Box(children=children, layout=self.box_layout)

        # Put the updated drop down in the output
        with output:
            display(self.current_drop_down)

        display(self.button_box, output)

    def on_normalize_button_clicked(self, toggle):
        if toggle.button_style == '':
            toggle.button_style = 'success'
            toggle.icon = 'check'
            self.normalize = True

        elif toggle.button_style == 'success':
            toggle.button_style = ''
            toggle.icon = ''
            self.normalize = False

    def on_done_button_clicked(self, b):
        """
        When the done button is clicked after the user had
        the option to define custom names this function is executed

        A load bar is displayed while te files are loaded, when the
        load bar is done the `.on_loaded_bar()` method is called.
        """
        clear_output(wait=True)

        display(self.load_bar)
        self.load_bar.observe(self.on_loaded_bar, names="value")

        self.load_bar.value += 1
        self.import_sound_files()

    def on_go_button_clicked(self, b):
        """
        Go button to display the analysis when all choices
        are made

        What happens :
        ___________________________________
        1. The output is cleared
        2. A output widget to store the output is instanciated
        3. The method in `self.analysis_tuple` is called
        4. The display is added to the output
        5. The 'Ok' button is enabled and the 'Go' button is disabled
        6. The drop down is set back to its default value
        7. The buttons and output are displayed
        """
        # Always clear the output
        clear_output(wait=True)
        output = widgets.Output(layout=self.out_layout)  # Create a output

        # Change the GUI state
        self.state = 'analysis displayed'

        # Case for a single sound
        if self.analysis == 'Single':

            # Case for Sound.plot_freq_bins method
            if self.analysis_tuple[0] == Sound.plot_freq_bins:
                # Call the method
                self.analysis_tuple[0](self.Sons, bins=[self.analysis_tuple[1]])

                # Define the title according to the chosen bin
                if self.analysis_tuple[1] == 'all':
                    plt.title('Frequency bin plot for ' + self.Sons.name)
                else:
                    plt.title(self.analysis_tuple[1] + ' bin plot for ' + self.Sons.name)
                # Add to output
                with output:
                    plt.show()

            # Case for the Sound.peak_damping method (print only)
            elif self.analysis_tuple[0] in [Sound.peak_damping, Sound.listen_freq_bins]:
                with output:
                    self.analysis_tuple[0](self.Sons)  # add print to output

            # Case for the Signal.plot method
            elif self.analysis_tuple[0] == Signal.plot:
                # Call the method according to normalization
                if not self.normalize:
                    self.analysis_tuple[0](self.Sons.signal, self.analysis_tuple[1])
                elif self.normalize:
                    self.analysis_tuple[0](self.Sons.signal.normalize(), self.analysis_tuple[1])
                # Define a title from the signal.plot(kind)
                plt.title(self.current_drop_down.label + ' for ' + self.Sons.name)
                # add to output
                with output:
                    plt.show()

            # Case for the Signal.listen method
            elif self.analysis_tuple[0] == Signal.listen:
                # add to output
                with output:
                    # Call the method according to normalization
                    if not self.normalize:
                        self.analysis_tuple[0](self.Sons.signal)
                    elif self.normalize:
                        self.analysis_tuple[0](self.Sons.signal.normalize())



        # Case for Dual and Multiple analyses
        elif self.analysis in ['Dual', 'Multiple']:

            # normalize the sound_pack if self.normalize is True
            if self.normalize:
                sound_pack = self.Pack.normalize()
            elif not self.normalize:
                sound_pack = self.Pack

            # Call with no arguments
            if len(self.analysis_tuple) == 1:
                # Case for a print output
                if self.display == 'print':
                    with output:
                        self.analysis_tuple[0](sound_pack)  # add print to output

                # Case for a plot output
                elif self.display == 'plot':
                    self.analysis_tuple[0](sound_pack)
                    with output:
                        plt.show()  # display plot in output

            # Call with arguments
            elif len(self.analysis_tuple) == 2:
                self.analysis_tuple[0](sound_pack, self.analysis_tuple[1])
                with output:
                    plt.show()

        # Setup the drop down to go back to method choice
        self.current_drop_down.value = 1
        self.current_drop_down = self.first_level_drop_down[self.analysis]
        self.current_drop_down.value = 1

        # Set the Go and Ok buttons to default value
        self.go_button.disabled = True
        self.ok_button.disabled = False

        # Set the normalization button to not normalized
        self.toggle_normalize_button.button_style = ''
        self.toggle_normalize_button.icon = ''
        self.normalize = False

        # display
        display(self.button_box, output)
        # Make the window larger
        display(HTML("<style>div.output_scroll { height: 44em; }</style>"))

    """
    Observe methods
    """

    def on_loaded_bar(self, change):
        """
        This method monitors the value of the load bar used
        when loading files.

        When the load bar is complete (value = 10), the
        button box is displayed with the "Ok" and "Go" buttons
        The "Go" button is disabled
        The Drop down with the methods according to the
        current analysis is displayed.
        """
        # When the bar reaches the end
        if change["new"] >= 10:
            clear_output(wait=True)

            # disable the go_button
            self.state = 'method choice'

            # Actualize the button box and display
            children = [self.ok_button, self.go_button, self.toggle_normalize_button]
            self.button_box = widgets.Box(children=children, layout=self.box_layout)
            self.ok_button.on_click(self.on_ok_button_clicked_2)
            self.go_button.on_click(self.on_go_button_clicked)
            self.toggle_normalize_button.on_click(self.on_normalize_button_clicked)
            self.go_button.disabled = True
            display(self.button_box)

            # create the output
            output = widgets.Output(layout=self.out_layout)

            # display the drop down associated to the current analysis
            self.current_drop_down = self.first_level_drop_down[self.analysis]
            with output:
                display(self.current_drop_down)

            display(output)

    """
    Back end functions
    """

    def define_sound_names(self):

        clear_output(wait=True)

        style = {'description_width': 'initial'}
        layout = widgets.Layout(width='40%')

        self.button_box = widgets.Box(children=[self.done_button], layout=self.box_layout)
        output = widgets.Output(layout=self.out_layout)
        display(self.button_box)

        with output:
            if self.analysis == 'Single':
                self.file_names = [ky for ky in self.single_file_selector.value.keys()]
                sound_name_input = widgets.Text(value='',
                                                placeholder='sound name',
                                                description=self.file_names[0],
                                                layout=layout,
                                                style=style,
                                                )
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
                                                    description=file,
                                                    layout=layout,
                                                    style=style,
                                                    )
                    display(sound_name_input)
                    self.sound_name_inputs.append(sound_name_input)

            elif self.analysis == 'Multiple':
                self.file_names = [ky for ky in self.mult_file_selector.value.keys()]
                self.sound_name_inputs = []
                for file in self.file_names:
                    sound_name_input = widgets.Text(value='',
                                                    placeholder='sound name',
                                                    description=file,
                                                    layout=layout,
                                                    style=style,
                                                    )
                    display(sound_name_input)
                    self.sound_name_inputs.append(sound_name_input)

        self.done_button.on_click(self.on_done_button_clicked)
        display(output)

    def import_sound_files(self):
        """
        Method to import the soundfile vectors into the program
        after the files and names where defined.
        *Only works with .wav files*
        """

        # Case for when only a single file is imported
        if self.analysis == 'Single':
            # Loading Bar Value = 0
            # Get the filename values from the file selector
            file_values = self.single_file_selector.value[self.file_names[0]]

            # Get the signal audio bytes
            bites = file_values['content']

            # Convert to wav audio object
            audio = wave.open(io.BytesIO(bites))

            sr = audio.getframerate()  # save the frame rate

            samples = []
            self.load_bar.value += 1  # LoadBar value = 1
            n = audio.getnframes()
            milestones = [int(i) for i in np.linspace(0, n, 5)][1:]
            for _ in range(audio.getnframes()):
                frame = audio.readframes(1)
                samples.append(struct.unpack("h", frame)[0])
                if _ in milestones:
                    self.load_bar.value += 1  # LoadBar value increases to 5 in loop

            self.load_bar.value += 1  # LoadBar value = 7
            signal = np.array(samples) / 32768
            Sound_Input = (signal, sr)
            self.load_bar.value += 1  # LoadBar value = 8
            if self.sound_name_inputs[0].value == '':
                name = self.file_names[0].replace('.wav', '')
            else:
                name = self.sound_name_inputs[0].value
            self.load_bar.value += 1  # LoadBar value = 9
            # This takes a long time
            self.Sons = Sound(Sound_Input, name=name).condition(return_self=True)
            self.load_bar.value += 2  # Loadbar = 10

        # Case for two files from two file selectors
        elif self.analysis == 'Dual':
            # LoadBar = 0
            self.Sons = []
            file_dicts = [self.dual_file_selector_1.value, self.dual_file_selector_2.value]
            self.load_bar.value += 2  # LoadBar = 1
            for file, dic, name_input in zip(self.file_names, file_dicts, self.sound_name_inputs):

                file_values = dic[file]
                bites = file_values['content']
                audio = wave.open(io.BytesIO(bites))
                sr = audio.getframerate()
                samples = []
                self.load_bar.value += 1  # LoadBar +=2
                for _ in range(audio.getnframes()):
                    frame = audio.readframes(1)
                    samples.append(struct.unpack("h", frame)[0])
                self.load_bar.value += 1  # LoadBar +=2
                signal = np.array(samples) / 32768
                Sound_Input = (signal, sr)
                if name_input.value == '':
                    name = file.replace('.wav', '')
                else:
                    name = name_input.value
                self.Sons.append(Sound(Sound_Input, name=name).condition(return_self=True))
                self.load_bar.value += 1  # LoadBar +=2
            # Load Bar = 8
            self.Pack = SoundPack(self.Sons, names=True)
            self.load_bar.value += 2  # Load Bar = 10

        # Case for multiple files
        # TODO : Better load bar using float
        elif self.analysis == 'Multiple':
            # LoadBar = 0
            self.Sons = []
            self.load_bar.value += 1  # LoadBar = 1

            for file, name_input in zip(self.file_names, self.sound_name_inputs):
                file_values = self.mult_file_selector.value[file]
                bites = file_values['content']
                audio = wave.open(io.BytesIO(bites))
                sr = audio.getframerate()
                samples = []
                for _ in range(audio.getnframes()):
                    frame = audio.readframes(1)
                    samples.append(struct.unpack("h", frame)[0])
                signal = np.array(samples) / 32768
                Sound_Input = (signal, sr)
                if name_input.value == '':
                    name = file.replace('.wav', '')
                else:
                    name = name_input.value
                self.Sons.append(Sound(Sound_Input, name=name).condition(return_self=True, verbose=False))
                if self.load_bar.value < 9:
                    self.load_bar.value += 1

            self.Pack = SoundPack(self.Sons, names=True)
            while self.load_bar.value < 10:
                self.load_bar.value += 1  # LoadBar = 10
