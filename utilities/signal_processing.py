from scipy.signal import butter, lfilter

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
