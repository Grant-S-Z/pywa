import torch
import numpy as np
from numba import jit
from scipy.special import erf
from math import erf as math_erf


def ser_waveform(x, amplitude, decay, sigma, t0):
    f_1 = amplitude / (2.0 * decay)
    f_2 = torch.exp(sigma**2 / (2.0 * decay**2) - (x - t0) / decay)
    arg = (decay * (x - t0) - sigma * sigma) / (2.0**0.5 * decay * sigma)
    f_3 = 1.0 + torch.erf(arg)
    return f_1 * f_2 * f_3


def ser_waveform_numpy(x: np.ndarray, amplitude: float, decay: float, sigma: float, t0: np.ndarray) -> np.ndarray:
    """
    Calculate SER waveform (numpy version).
    Supports broadcasting for t0.
    x: (W,)
    t0: (N,) or scalar
    Returns: (W, N) or (W,)
    """
    # Ensure x is column vector if t0 is array to broadcast
    if isinstance(t0, np.ndarray) and t0.ndim > 0:
        x = x[:, np.newaxis]
        # t0 is (N,)
        # Result will be (W, N)
    
    f_1 = amplitude / (2.0 * decay)
    # exp argument
    exp_arg = sigma**2 / (2.0 * decay**2) - (x - t0) / decay
    f_2 = np.exp(exp_arg)
    
    # erf argument
    erf_arg = (decay * (x - t0) - sigma * sigma) / (np.sqrt(2.0) * decay * sigma)
    f_3 = 1.0 + erf(erf_arg)
    
    return f_1 * f_2 * f_3


@jit(nopython=True, fastmath=True)
def ser_waveform_jit(x, amplitude, decay, sigma, t0):
    """Numba-accelerated SER waveform calculation"""
    result = np.zeros((len(x), len(t0)))
    for i in range(len(x)):
        for j in range(len(t0)):
            f_1 = amplitude / (2.0 * decay)
            exp_arg = sigma**2 / (2.0 * decay**2) - (x[i] - t0[j]) / decay
            f_2 = np.exp(exp_arg)
            erf_arg = (decay * (x[i] - t0[j]) - sigma * sigma) / (np.sqrt(2.0) * decay * sigma)
            f_3 = 1.0 + math_erf(erf_arg)
            result[i, j] = f_1 * f_2 * f_3
    return result
