from IPython.display import display, clear_output, HTML
from IPython import get_ipython
from guitarsounds.analysis import Plot, Signal, Sound, SoundPack
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

    # Fundamental input style
    fundamental_style = style = {'description_width': 'initial'}

    # Attribute for output layout
    output = widgets.Output(layout={'border': '1px solid black'})

    # List of plot methods
    plot_methods = [Plot.signal, Plot.envelop, Plot.log_envelop, Plot.fft, Plot.fft_hist, Plot.peaks, Plot.peak_damping,
                    Plot.time_damping, ]
    bin_ticks_methods = [Plot.fft, Plot.fft_hist, Plot.peaks, Plot.peak_damping, ]

    # Plot info dict
    plot_info_dict = {'signal': Plot.signal,
                      'envelop': Plot.envelop,
                      'log envelop': Plot.log_envelop,
                      'fft': Plot.fft,
                      'fft hist': Plot.fft_hist,
                      'peaks': Plot.peaks,
                      'peak damping': Plot.peak_damping,
                      'time damping': Plot.time_damping,
                      'integral': Plot.integral}

    # analysis dropdowns
    # Single analysis drop down
    options = [('', 1),
               ('Listen Sound', Signal.listen),
               ('Listen frequency bins', Sound.listen_freq_bins),
               ('Frequency bin plot', Sound.plot_freq_bins),
               ('Frequency bin histogram', Sound.bin_hist),
               ('Signal plot', Plot.signal),
               ('Envelop plot', Plot.envelop),
               ('Log-envelop plot', Plot.log_envelop),
               ('Fourier transform plot', Plot.fft),
               ('Fourier transform histogram', Plot.fft_hist),
               ('Peaks plot', Plot.peaks),
               ('Peak damping plot', Plot.peak_damping),
               ('Time damping plot', Plot.time_damping),
               ('Frequency damping values', Sound.peak_damping), ]

    drop_down_style = {'description_width': '150px'}
    single_drop_down = widgets.Dropdown(options=options, value=1, style=drop_down_style,
                                        description='Choose an analysis : ')
    single_drop_down.rank = 'first'

    unique_plot_methods = [SoundPack.compare_peaks, SoundPack.fft_mirror, SoundPack.fft_diff, SoundPack.plot,
                           SoundPack.bin_power_hist, ]

    # Dual analysis drop down
    options = [('', 1),
               ('Compare Peaks', SoundPack.compare_peaks),
               ('Mirror FFT', SoundPack.fft_mirror),
               ('FFT difference', SoundPack.fft_diff),
               ('Bin power comparison', SoundPack.integral_compare),
               ('Stacked plot', SoundPack.plot),
               ('Compared plot', SoundPack.compare_plot),
               ('Bin power plot', SoundPack.integral_plot),
               ('Bin power table', SoundPack.bin_power_table),
               ('Bin power histogram', SoundPack.bin_power_hist),
               ('Frequency Bin plot', SoundPack.freq_bin_plot),
               ('Print Fundamentals', SoundPack.fundamentals), ]

    dual_drop_down = widgets.Dropdown(options=options, value=1, style=drop_down_style,
                                      description='Choose an analysis : ')
    dual_drop_down.rank = 'first'

    # Multiple analysis drop down
    options = [('', 1),
               ('Stacked plot', SoundPack.plot),
               ('Compared plot', SoundPack.compare_plot),
               ('Frequency Bin plot', SoundPack.freq_bin_plot),
               ('Combine Envelops', SoundPack.combine_envelop),
               ('Print Fundamentals', SoundPack.fundamentals),
               ('Bin power plot', SoundPack.integral_plot),
               ('Print bin powers', SoundPack.bin_power_table),
               ('Bin power histogram', SoundPack.bin_power_hist), ]

    DM_bin_choice_methods = [SoundPack.freq_bin_plot, SoundPack.integral_plot, SoundPack.integral_compare]

    mult_drop_down = widgets.Dropdown(options=options, value=1, style=drop_down_style,
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
               ('brillance', 'brillance'), ]

    bin_drop_down = widgets.Dropdown(options=options, value='all', style=drop_down_style,
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
               ('Timbre Attributes', 'timbre'),
               ('Cumulative integral', 'integral'), ]

    plot_drop_down = widgets.Dropdown(options=options, value='signal', style=drop_down_style,
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

        # Info button
        self.info_button = widgets.Button(description='Info')

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

        # Dict with dropdown methods to display the menu associated to
        # the analysis
        self.first_level_drop_down = {'Single': self.single_drop_down,
                                      'Dual': self.dual_drop_down,
                                      'Multiple': self.mult_drop_down}

        # Initiate name spaces
        self.analysis = None
        self.display = None
        self.current_drop_down = None
        self.Pack = None
        self.analysis_tuple = None
        self.file_names = None
        self.sound_name_inputs = None
        self.sound_fundamental_inputs = None
        self.sounds = None

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
        if b is not None:
            pass
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
        if b is not None:
            pass
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
        if b is not None:
            pass
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
        The user clicks this button when he is done choosing files and when
        he is done defining names
        """
        if b is not None:
            pass
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

        # If the file were chosen the user is taken to the "define name" interface
        if files_where_chosen:
            self.define_sound_names()

        # if not we go back to file selection
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

                # Display an error if a file selector was clicked but no file was chosen
                if self.analysis in ['Single', 'Dual', 'Multiple']:
                    error = generate_error_widget('No sound was chosen')
                    display(error)

            display(self.button_box)
            display(output)

    """ 
    Analysis interface button click methods
    """

    def on_ok_button_clicked_2(self, b):
        """
        Method to make the "Ok" button interact with the
        analysis method choice.

        __ when interface.state = 'method choice' __
        - The "Ok" and "Go" buttons appears after the loading bar is done
        - The dropdown corresponds to the methods associated to
        the analysis
        """
        if b is not None:
            pass
        # Clear the Output
        clear_output(wait=True)
        output = widgets.Output(layout=self.out_layout)

        # Save the dropdown value
        drop_down_value = self.current_drop_down.value
        
        # enable the info button when coming back from display
        if self.state != 'display':
            self.info_button.disabled = False
            self.toggle_normalize_button.disabled = False

        # Deactivate the info button if it was activated
        if self.info_button.button_style == 'info':
            self.info_button.button_style = ''

        if self.state == 'method choice':  # State when the user is choosing the analysis method

            # If we only analyse a single sound
            if self.analysis == 'Single':

                # Special case when the method is the frequency bin plot
                if drop_down_value == Sound.plot_freq_bins:
                    self.analysis_tuple = [drop_down_value]  # Store the method
                    # Change the dropdown to frequency bin choice
                    self.current_drop_down = self.bin_drop_down
                    self.state = 'method choice 2'  # a second choice is needed
                    self.display = 'plot'

                # Case for the methods without plotting
                elif drop_down_value in [Sound.peak_damping, Sound.listen_freq_bins, Signal.listen]:
                    self.analysis_tuple = [drop_down_value]  # store the method
                    self.state = 'display'  # ready to display
                    self.display = 'print'

                # Signal.plot.method() methods
                elif drop_down_value in [*self.plot_methods, Sound.bin_hist]:
                    # store method and arg in a list
                    self.analysis_tuple = [drop_down_value]
                    self.state = 'display'  # ready to display
                    self.display = 'plot'

                # Error when no method is chosen
                elif drop_down_value == 1:
                    error = generate_error_widget('No analysis method was chosen')
                    with output:
                        display(error)

            # Case when two sounds or multiple sounds are being analysed
            elif self.analysis in ['Dual', 'Multiple']:

                # Special case for the frequency bin plot
                if drop_down_value in self.DM_bin_choice_methods:
                    self.analysis_tuple = [drop_down_value]  # Store the method
                    # Update the dropdown to frequency bin choice
                    self.current_drop_down = self.bin_drop_down
                    self.state = 'method choice 2'  # a second choice is needed
                    self.display = 'plot'

                # Case for plot methods
                elif (drop_down_value == SoundPack.plot) or (drop_down_value == SoundPack.compare_plot):
                    self.analysis_tuple = [drop_down_value]  # Store the method
                    # Update the dropdown to the plot dropdown
                    self.current_drop_down = self.plot_drop_down
                    self.state = 'method choice 2'  # a second choice is needed
                    self.display = 'plot'

                # Error when no method is chosen
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

        # if we are coming back from the display the state is redefined, and we restart
        elif self.state == 'analysis displayed':
            self.state = 'method choice'

        # If the button is pressed and the method is defined, the go button is enabled
        if self.state == 'display':
            self.go_button.disabled = False
            self.ok_button.disabled = True
            self.info_button.disabled = True
            self.toggle_normalize_button.disabled = True

        # Actualize the button box and display
        children = [self.ok_button, self.go_button, self.toggle_normalize_button, self.info_button]
        self.button_box = widgets.Box(children=children, layout=self.box_layout)

        # Put the updated dropdown in the output
        with output:
            display(self.current_drop_down)

        display(self.button_box, output)

    def on_info_button_clicked(self, info):
        """
        Method called when the info button is clicked
        Displays the help string associated with the current dropdown method
        """
        if info.button_style == '':
            # change the style to make the button blue
            info.button_style = 'info'

            # Clear the Output
            clear_output(wait=True)
            output = widgets.Output(layout=self.out_layout)

            # Case when the user is selecting the first method
            if self.state == 'method choice':

                # if the method is a tuple with an argument
                if type(self.current_drop_down.value) == tuple:
                    with output:
                        display(help(self.current_drop_down.value[0]))

                # if no method was selected
                elif type(self.current_drop_down.value) == int:
                    error = generate_error_widget('No analysis was selected')
                    with output:
                        display(error)

                # a method not in a tuple was selected
                else:
                    with output:
                        display(help(self.current_drop_down.value))

                # display every thing
                display(self.button_box, output)

            # case when the user is doing a secondary selection
            elif self.state == 'method choice 2':

                # case for the plot type dropdown
                if self.current_drop_down.name == 'plot':
                    with output:
                        display(help(self.plot_info_dict[self.current_drop_down.value]))

                # case for bin type dropdown (display the previous method)
                elif self.current_drop_down.name == 'bin':
                    with output:
                        display(help(self.analysis_tuple[0]))

                display(self.button_box, output)

        elif info.button_style == 'info':
            info.button_style = ''

            # Clear the Output
            clear_output(wait=True)
            output = widgets.Output(layout=self.out_layout)
            with output:
                display(self.current_drop_down)

            # display every thing
            display(self.button_box, output)

    def on_normalize_button_clicked(self, toggle):
        """
        Method called when the normalize button is clicked
        The normalized attribute is inverted according to the current value
        """
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
        if b is not None:
            pass
        clear_output(wait=True)

        display(self.load_bar)
        self.load_bar.observe(self.on_loaded_bar, names="value")

        self.load_bar.value += 1
        self.import_sound_files()

    def on_go_button_clicked(self, b):
        """
        Go button to display the analysis when all choices are made

        What happens :
        ___________________________________
        1. The output is cleared
        2. An output widget to store the output is instanced
        3. The method in `self.analysis_tuple` is called
        4. The display is added to the output
        5. The 'Ok' button is enabled and the 'Go' button is disabled
        6. The dropdown is set back to its default value
        7. The buttons and output are displayed
        """
        if b is not None:
            pass
        # Always clear the output
        clear_output(wait=True)
        output = widgets.Output(layout=self.out_layout)  # Create a output

        # Change the GUI state
        self.state = 'analysis displayed'

        # Set the matplotlib display method
        get_ipython().run_line_magic('matplotlib', 'inline')

        # Case for a single sound
        if self.analysis == 'Single':

            # Case for Sound.plot_freq_bins method
            if self.analysis_tuple[0] == Sound.plot_freq_bins:
                # change interface
                get_ipython().run_line_magic('matplotlib', 'notebook')
                # create a figure
                plt.figure(figsize=(8, 6))
                # Call the method
                self.analysis_tuple[0](self.sounds, bins=[self.analysis_tuple[1]])

                # Define the title according to the chosen bin
                if self.analysis_tuple[1] == 'all':
                    plt.title('Frequency bin plot for ' + self.sounds.name)
                else:
                    plt.title(self.analysis_tuple[1] + ' bin plot for ' + self.sounds.name)

                    plt.show()

            # Case for the Sound.peak_damping method (print only)
            elif self.analysis_tuple[0] in [Sound.peak_damping, Sound.listen_freq_bins]:
                with output:
                    self.analysis_tuple[0](self.sounds)  # add print to output

            # Case for the Signal.plot method
            elif self.analysis_tuple[0] in self.plot_methods:
                # change plot interface
                get_ipython().run_line_magic('matplotlib', 'notebook')
                # create a figure
                plt.figure(figsize=(8, 6))
                # Add the fill argument if there is just one plot
                kwargs = {}
                # Call the method according to normalization
                if not self.normalize:
                    self.analysis_tuple[0](self.sounds.signal.plot, **kwargs)
                elif self.normalize:
                    self.analysis_tuple[0](self.sounds.signal.normalize().plot, **kwargs)

                if self.analysis_tuple[0] == Plot.time_damping:
                    zeta = np.around(self.sounds.signal.time_damping(), 5)
                    plt.title(self.current_drop_down.label + ' for ' + self.sounds.name + ' Zeta = ' + str(zeta))
                # Define a title from the signal.plot(kind)
                else:
                    plt.title(self.current_drop_down.label + ' for ' + self.sounds.name)

                # make the x-axis ticks the frequency bins if the axe is frequency
                if self.analysis_tuple[0] in self.bin_ticks_methods:
                    Plot.set_bin_ticks(self.sounds.signal.plot)
                # add to output
                with output:
                    plt.show()

            # Case for the Sound.bin_hist method
            elif self.analysis_tuple[0] == Sound.bin_hist:
                # change plot interface
                get_ipython().run_line_magic('matplotlib', 'notebook')
                # call the method
                self.analysis_tuple[0](self.sounds)
                # set a title
                plt.title(self.current_drop_down.label + ' for ' + self.sounds.name)
                # add to output
                with output:
                    plt.show()

            # Case for the Signal.listen method
            elif self.analysis_tuple[0] == Signal.listen:
                # add to output
                with output:
                    # Call the method according to normalization
                    if not self.normalize:
                        self.analysis_tuple[0](self.sounds.signal)
                    elif self.normalize:
                        self.analysis_tuple[0](self.sounds.signal.normalize())

        # Case for Dual and Multiple analyses
        elif self.analysis in ['Dual', 'Multiple']:

            # normalize the sound_pack if self.normalize is True
            if self.normalize:
                sound_pack = self.Pack.normalize()
            else:
                sound_pack = self.Pack

            # if the analysis method is a unique plot, make matplotlib interactive
            get_ipython().run_line_magic('matplotlib', 'inline')
            if self.analysis_tuple[0] in self.unique_plot_methods:
                get_ipython().run_line_magic('matplotlib', 'notebook')

            # Call with no arguments
            if len(self.analysis_tuple) == 1:
                # Case for a print output
                if self.display == 'print':
                    with output:
                        self.analysis_tuple[0](sound_pack)  # add print to output

                # special case to have bins ticks for the fft_diff method
                elif self.analysis_tuple[0] == SoundPack.fft_diff:
                    self.analysis_tuple[0](sound_pack, ticks='bins')
                    with output:
                        plt.show()  # display plot in output

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

        # Set up the dropdown to go back to method choice
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
        The dropdown with the methods according to the
        current analysis is displayed.
        """
        # When the bar reaches the end
        if change["new"] >= 10:
            clear_output(wait=True)

            # disable the go_button
            self.state = 'method choice'

            # Actualize the button box and display
            children = [self.ok_button, self.go_button, self.toggle_normalize_button, self.info_button]
            self.button_box = widgets.Box(children=children, layout=self.box_layout)
            self.ok_button.on_click(self.on_ok_button_clicked_2)
            self.go_button.on_click(self.on_go_button_clicked)
            self.toggle_normalize_button.on_click(self.on_normalize_button_clicked)
            self.info_button.on_click(self.on_info_button_clicked)
            self.go_button.disabled = True
            display(self.button_box)

            # create the output
            output = widgets.Output(layout=self.out_layout)

            # display the dropdown associated to the current analysis
            self.current_drop_down = self.first_level_drop_down[self.analysis]
            with output:
                display(self.current_drop_down)

            display(output)

    """
    Back end functions
    """

    def define_sound_names(self):
        """
        A method to define sound names and fundamentals
        """

        # Clear the output and define the new one
        clear_output(wait=True)
        output = widgets.Output(layout=self.out_layout)

        # Style for the text inputs
        style = {'description_width': 'initial'}

        # Small string 'Hz' to indicate units
        HZ_string = widgets.HTML('<p>' + 'Hz' + '</p>')

        # Define the button box
        self.button_box = widgets.Box(children=[self.done_button], layout=self.box_layout)

        # Define the output with the text inputs
        with output:

            # Case for a single sound analysis
            if self.analysis == 'Single':

                # get the filenames
                self.file_names = [ky for ky in self.single_file_selector.value.keys()]

                # make a sound name input widget
                sound_name_input = widgets.Text(value='',
                                                placeholder='sound name',
                                                description=self.file_names[0],
                                                layout=widgets.Layout(width='40%'),
                                                style=style,
                                                )

                # make a fundamental input widget
                fundamental_input = widgets.FloatText(value=0,
                                                      description='Fundamental :',
                                                      style=style,
                                                      layout=widgets.Layout(width='20%')
                                                      )

                # children that go in the name box
                children = [sound_name_input, fundamental_input, HZ_string]

                # define a name box widget
                name_box_layout = widgets.Layout(align_items='stretch', flex_flow='line', width='75%')
                name_box = widgets.Box(children=children, layout=name_box_layout)

                # display the box
                display(name_box)

                # store the input to refer them later
                self.sound_name_inputs = [sound_name_input]
                self.sound_fundamental_inputs = [fundamental_input]

            # Case for dual sound analysis
            elif self.analysis in ['Dual', 'Multiple']:

                if self.analysis == 'Dual':
                    # get the file names
                    name1 = [ky for ky in self.dual_file_selector_1.value.keys()][0]
                    name2 = [ky for ky in self.dual_file_selector_2.value.keys()][0]
                    self.file_names = [name1, name2]

                elif self.analysis == 'Multiple':
                    self.file_names = [ky for ky in self.mult_file_selector.value.keys()]

                # create empty lists for the inputs
                self.sound_name_inputs = []
                self.sound_fundamental_inputs = []

                for file in self.file_names:
                    # make a text input widget
                    sound_name_input = widgets.Text(value='',
                                                    placeholder='sound name',
                                                    description=file,
                                                    layout=widgets.Layout(width='40%'),
                                                    style=style,
                                                    )

                    # make a fundamental input widget
                    fundamental_input = widgets.FloatText(value=0,
                                                          description='Fundamental :',
                                                          layout=widgets.Layout(width='20%'),
                                                          style=style
                                                          )

                    # children that go in the name box
                    children = [sound_name_input, fundamental_input, HZ_string]

                    # define a name box widget
                    name_box_layout = widgets.Layout(align_items='stretch', flex_flow='line', width='75%')
                    name_box = widgets.Box(children=children, layout=name_box_layout)

                    # display the box
                    display(name_box)

                    # append the inputs
                    self.sound_name_inputs.append(sound_name_input)
                    self.sound_fundamental_inputs.append(fundamental_input)

        self.done_button.on_click(self.on_done_button_clicked)

        # display everything
        display(self.button_box, output)

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

            # Get the sound name
            if self.sound_name_inputs[0].value == '':
                name = self.file_names[0].replace('.wav', '')
            else:
                name = self.sound_name_inputs[0].value

            # Get the sound fundamental
            if self.sound_fundamental_inputs[0].value == 0:
                fundamental = None
            else:
                fundamental = self.sound_fundamental_inputs[0].value

            self.load_bar.value += 1  # LoadBar value = 9
            # This takes a long time
            sound = Sound(Sound_Input, name=name, fundamental=fundamental)
            self.sounds = sound.condition(return_self=True, verbose=False)
            self.load_bar.value += 2  # Load-bar = 10

        # Case for two files from two file selectors
        elif self.analysis == 'Dual':
            # LoadBar = 0
            self.sounds = []
            file_dicts = [self.dual_file_selector_1.value, self.dual_file_selector_2.value]
            self.load_bar.value += 2  # LoadBar = 1

            # zipped iterator
            iterator = zip(self.file_names, file_dicts, self.sound_name_inputs, self.sound_fundamental_inputs)

            #  Create a sound for every file
            for file, dic, name_input, fundamental_input in iterator:

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

                # get the name value
                if name_input.value == '':
                    name = file.replace('.wav', '')
                else:
                    name = name_input.value

                # get the fundamental value
                if fundamental_input.value == 0:
                    fundamental = None
                else:
                    fundamental = fundamental_input.value

                sound = Sound(Sound_Input, name=name, fundamental=fundamental)
                sound.condition(verbose=False)
                self.sounds.append(sound)
                self.load_bar.value += 1  # LoadBar +=2
            # Load Bar = 8
            self.Pack = SoundPack(self.sounds, names=[sound.name for sound in self.sounds])
            self.load_bar.value += 2  # Load Bar = 10

        # Case for multiple files
        elif self.analysis == 'Multiple':
            # LoadBar = 0
            self.sounds = []
            self.load_bar.value += 1  # LoadBar = 1

            # zipped iterator
            iterator = zip(self.file_names, self.sound_name_inputs, self.sound_fundamental_inputs)

            for file, name_input, fundamental_input in iterator:
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

                # get the sound names
                if name_input.value == '':
                    name = file.replace('.wav', '')
                else:
                    name = name_input.value

                # get the fundamental values
                if fundamental_input.value == 0:
                    fundamental = None
                else:
                    fundamental = fundamental_input.value

                sound = Sound(Sound_Input, name=name, fundamental=fundamental)
                sound.condition(verbose=False)
                self.sounds.append(sound)
                if self.load_bar.value < 9:
                    self.load_bar.value += 1

            self.Pack = SoundPack(self.sounds, names=[sound.name for sound in self.sounds])
            while self.load_bar.value < 10:
                self.load_bar.value += 1  # LoadBar = 10
