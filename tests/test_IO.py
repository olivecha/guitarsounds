import unittest
import wave
import guitarsounds
import librosa
from guitarsounds.helpers_tests import get_rnd_audio_file


class MyTestCase(unittest.TestCase):

    def test_wave_read(self):
        """ Test reading a wave file with the wave package"""
        file = get_rnd_audio_file()
        signal, sr = librosa.load(file)
        WavObject = wave.open(file)
        sr_wav = WavObject.getframerate()
        self.assertAlmostEqual(sr_wav, sr)


if __name__ == '__main__':
    unittest.main()
