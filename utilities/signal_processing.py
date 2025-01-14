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

def extract_fifth_harmonic1(signal, sampling_rate):
    """
    Extract the fifth harmonic from the given signal using FFT.

    Parameters:
        signal (numpy.ndarray): Input signal array (1D or 2D).
            For 2D, each row is considered an individual signal.
        sampling_rate (int): Sampling rate of the signal in Hz.

    Returns:
        numpy.ndarray: The fifth harmonic component of the input signal, with the same shape as the input.
    """
    # Ensure signal is a 2D array for consistent processing
    signal = np.atleast_2d(signal)

    # Number of samples
    n_samples = signal.shape[1]

    # Frequency resolution
    freq_resolution = sampling_rate / n_samples

    # Frequency indices corresponding to harmonics
    fifth_harmonic_index = int(5 * n_samples / sampling_rate)

    # FFT of the signal
    fft_signal = np.fft.fft(signal, axis=1)

    # Zero out all frequencies except the fifth harmonic and its conjugate
    filtered_fft = np.zeros_like(fft_signal)
    filtered_fft[:, fifth_harmonic_index] = fft_signal[:, fifth_harmonic_index]
    filtered_fft[:, -fifth_harmonic_index] = fft_signal[:, -fifth_harmonic_index]

    # Inverse FFT to get back to time domain
    fifth_harmonic = np.fft.ifft(filtered_fft, axis=1).real

    # If input was 1D, return a 1D array
    if signal.shape[0] == 1:
        return fifth_harmonic.flatten()

    return fifth_harmonic
    
def extract_fifth_harmonic2(signal, sampling_rate):
    """
    Extracts the fifth-order harmonic component of a 1D array signal.
    
    Parameters:
        signal (numpy.ndarray): 1D array representing the input signal.
        sampling_rate (float): Sampling rate of the signal in Hz.
    
    Returns:
        numpy.ndarray: The fifth-order harmonic component of the signal.
    """
    # Perform FFT on the signal
    fft_result = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(signal), d=1/sampling_rate)
    
    # Find the fundamental frequency
    magnitude = np.abs(fft_result)
    fundamental_idx = np.argmax(magnitude[1:]) + 1  # Exclude the DC component at index 0
    fundamental_freq = frequencies[fundamental_idx]
    
    # Identify the fifth harmonic frequency
    fifth_harmonic_freq = 5 * fundamental_freq
    
    # Create a mask to isolate the fifth harmonic
    fifth_harmonic_mask = np.isclose(np.abs(frequencies), fifth_harmonic_freq, atol=fundamental_freq/2)
    
    # Filter out all other frequencies except the fifth harmonic
    filtered_fft = np.zeros_like(fft_result)
    filtered_fft[fifth_harmonic_mask] = fft_result[fifth_harmonic_mask]
    
    # Perform the inverse FFT to reconstruct the time-domain fifth harmonic signal
    fifth_harmonic_signal = np.fft.ifft(filtered_fft).real
    
    return fifth_harmonic_signal
