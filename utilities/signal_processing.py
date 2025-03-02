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

def extract_fifth_harmonic(signal, fs):
    N = len(signal)
    freqs = np.fft.fftfreq(N, d=1/fs)  # Frequency bins
    fft_values = np.fft.fft(signal)  # Compute FFT

    # Find the index corresponding to the 5th harmonic
    fundamental_freq = fs / N  # Fundamental frequency
    fifth_harmonic_freq = 5 * fundamental_freq
    idx = np.argmin(np.abs(freqs - fifth_harmonic_freq))

    # Extract the 5th harmonic component
    fifth_harmonic = 2 * np.abs(fft_values[idx]) / N
    phase = np.angle(fft_values[idx])

    return fifth_harmonic, phase

def compute_5th_harmonic_zero_sequence(voltage_signals, current_signals, fs):
    # Extract 5th harmonic components for voltages and currents
    voltage_5th_harmonics = [extract_fifth_harmonic(voltage_signals[:, i], fs) for i in range(3)]
    current_5th_harmonics = [extract_fifth_harmonic(current_signals[:, i], fs) for i in range(3)]

    # Compute zero-sequence components (V0 = (Va + Vb + Vc) / 3, I0 = (Ia + Ib + Ic) / 3)
    V0_5th = sum([v[0] * np.exp(1j * v[1]) for v in voltage_5th_harmonics]) / 3
    I0_5th = sum([i[0] * np.exp(1j * i[1]) for i in current_5th_harmonics]) / 3

    # Get magnitude and phase of zero-sequence components
    return {
        "V0_magnitude": np.abs(V0_5th),
        "V0_phase": np.angle(V0_5th),
        "I0_magnitude": np.abs(I0_5th),
        "I0_phase": np.angle(I0_5th)
    }
