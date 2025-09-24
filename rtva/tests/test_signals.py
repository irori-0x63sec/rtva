import numpy as np
from rtva.dsp.pitch import yin_f0

def test_yin_sine_440():
    sr = 44100
    t = np.arange(0, 0.1, 1/sr)
    x = np.sin(2*np.pi*440*t).astype(np.float32)
    f0 = yin_f0(x, sr, 75, 900)
    assert abs(f0 - 440) < 5
