from scipy.signal import butter, lfilter, bessel
import numpy as np

def high_pass_filter(data, cutoff, fs, order=4):
    """
    Applies a high-pass Butterworth filter to the input data.
    
    Parameters:
    - data: Input signal (1D array).
    - cutoff: Cutoff frequency of the filter (Hz).
    - fs: Sampling frequency (Hz).
    - order: Filter order (default: 4).
    
    Returns:
    - Filtered signal (1D array).
    """
    nyquist = 0.5 * fs  # Nyquist frequency (half the sampling frequency)
    normalized_cutoff = cutoff / nyquist  # Normalize cutoff relative to Nyquist
    b, a = butter(order, normalized_cutoff, btype='high', analog=False)  # Filter coefficients
    return lfilter(b, a, data)  # Apply the filter to the input signal

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')


def band_pass_filter(data, low_cutoff, high_cutoff, fs, order=4, filter_type='butter', smooth=False):
    """
    Band-pass filter with options for filter type and smoothing.
    
    Parameters:
    - data: Input signal (1D array).
    - low_cutoff: Low cutoff frequency (Hz).
    - high_cutoff: High cutoff frequency (Hz).
    - fs: Sampling frequency (Hz).
    - order: Filter order (default: 4).
    - filter_type: Type of filter ('butter', 'bessel').
    - smooth: Apply moving average smoothing (default: False).
    
    Returns:
    - Filtered signal (1D array).
    """
    nyquist = 0.5 * fs
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    
    # Choose filter type
    if filter_type == 'butter':
        b, a = butter(order, [low, high], btype='band', analog=False)
    elif filter_type == 'bessel':
        b, a = bessel(order, [low, high], btype='band', analog=False)
    else:
        raise ValueError("Unsupported filter type. Use 'butter' or 'bessel'.")
    
    filtered_data = lfilter(b, a, data)
    
    # Optional smoothing
    if smooth:
        filtered_data = moving_average(filtered_data, window_size=10)
    
    return filtered_data
