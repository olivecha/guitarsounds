# __init__.py
from guitarsounds.interface import guitarGUI as Interface
from guitarsounds.analysis import Signal, Sound, SoundPack, Plot
from guitarsounds.analysis import plt
import guitarsounds.helpers_tests
show = plt.show

# Version of the guitarsounds package
__version__ = '1.0.0'
