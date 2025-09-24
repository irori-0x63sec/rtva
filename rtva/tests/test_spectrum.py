import numpy as np

from rtva.dsp.spectrum import stft_mag_db


def test_stft_mag_db_shapes():
    sr = 8000
    hop = 160
    frames = np.random.randn(5, 320).astype(np.float32)
    spec_db, freq_axis, time_axis = stft_mag_db(frames, sr, n_fft=512, hop=hop)

    assert spec_db.shape == (257, 5)
    assert freq_axis.shape[0] == 257
    assert time_axis.shape[0] == 5
    assert np.isclose(time_axis[0], -(4 * hop) / sr)
    assert np.isclose(time_axis[-1], 0.0)
