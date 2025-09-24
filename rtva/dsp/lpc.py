import numpy as np
from scipy.signal import lfilter

def pre_emphasis(x: np.ndarray, coef=0.97):
    return lfilter([1, -coef], [1], x)
