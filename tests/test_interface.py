import unittest
from guitarsounds import interface, Sound
from helpers_tests import get_rnd_audio_file
import numpy as np
import ipywidgets


class MyTestCase(unittest.TestCase):
    """ class for the inteface unit tests"""

    def test_generate_error_widget(self):
        """
        test the generate_error_widget function from the interface module
        """
        self.assertIsInstance(interface.generate_error_widget('string'),
                              ipywidgets.widgets.widget_string.HTML)

    def test_GUI_initialization(self):
        """ test the initialization of the interface GUI class """
        self.assertIsInstance(interface.guitarGUI(), interface.guitarGUI)

    def test_on_single_button_clicked(self):
        """ test the on_single_button_clicked method of the guitarGUI class """
        gui = interface.guitarGUI()
        gui.on_single_button_clicked(None)
        # If this is reached the test is passed
        self.assertTrue(True)

    def test_on_dual_button_clicked(self):
        """ test the on_dual_button_clicked method of the guitarGUI """
        gui = interface.guitarGUI()
        gui.on_dual_button_clicked(None)
        # If this is reached the test is passed
        self.assertTrue(True)

    def test_on_multiple_button_clicked(self):
        """ test the on_multiple_button_clicked method of the guitarGUI """
        gui = interface.guitarGUI()
        gui.on_multiple_button_clicked(None)
        # If this is reached the test is passed
        self.assertTrue(True)

    def test_change_file_selection_state(self):
        """ test the file selection state changer of the guitarGUI """
        gui = interface.guitarGUI()
        # test both branches
        gui.change_file_selection_state(False)
        gui.change_file_selection_state(True)
        # If this is reached the test is passed
        self.assertTrue(True)

    def test_on_ok_button_clicked_1(self):
        """ test the on_ok_button_clicked_1 method of the guitarGUI """
        gui = interface.guitarGUI()
        # test both branches
        gui.on_ok_button_clicked_1(None)
        # If this is reached the test is passed
        self.assertTrue(True)

    def test_on_ok_button_clicked_2(self):
        """ test the on_ok_button_clicked_2 method of the guitarGUI """
        # instantiate the interface
        gui = interface.guitarGUI()
        # Test all the possible branches
        for state in ['method choice', 'method choice 2', 'display']:
            for case in ['Single', 'Dual', 'Multiple']:
                gui.state = state
                gui.analysis_tuple = [Sound.plot_freq_bins]
                gui.current_drop_down = gui.first_level_drop_down[case]
                gui.on_ok_button_clicked_2(None)

        # if this is reached the test pass
        self.assertTrue(True)

    def test_on_info_button_clicked(self):
        """ test the on_info_button_clicked method of the guitarGUI """
        gui = interface.guitarGUI()
        info_button = ipywidgets.Button(description='Info')
        gui.on_info_button_clicked(info_button)
        self.assertTrue(True)

    def test_on_normalize_button_clicked(self):
        """ test the on_normalize_button_clicked method of the guitarGUI """
        gui = interface.guitarGUI()
        toggle_normalize_button = ipywidgets.Button(description='Normalize')
        gui.on_info_button_clicked(toggle_normalize_button)
        self.assertTrue(True)

    def test_on_done_button_clicked(self):
        """ test the on_done_button_clicked """
        gui = interface.guitarGUI()
        gui.on_done_button_clicked(None)
        self.assertTrue(True)

    def test_on_go_button_clicked(self):
        """ test the on_go_button_clicked method of the guitarGUI """
        # The go button only works when an ipython interface is available
        pass

    def test_on_loaded_bar(self):
        """ test the on_loaded_bar method of the guitarGUI """
        gui = interface.guitarGUI()
        change = {'new':11}
        for analysis in ['Single', 'Dual', 'Multiple']:
            gui.analysis = analysis
            gui.on_loaded_bar(change)
        self.assertTrue(True)

    def test_define_sound_names(self):
        """ test the define_sound_names method of the guitarGUI """
        gui = interface.guitarGUI()
        for analysis in ['Single', 'Dual', 'Multiple']:
            gui.define_sound_names()
        self.assertTrue(True)

    def test_import_sound_files(self):
        """ test the import_sound_files method of the guitarGUI """
        gui = interface.guitarGUI()
        for analysis in ['Single', 'Dual', 'Multiple']:
            gui.import_sound_files()
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
