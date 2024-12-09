import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import pyComtrade

# High-pass filter function (same as before)
def high_pass_filter(data, cutoff, fs, order=4):
    """
    Applies a high-pass Butterworth filter to the input data.
    
    Parameters:
    - data: Input signal (1D array).
    - cutoff: Cutoff frequency of the filter (Hz).
    - fs: Sampling frequency (Hz).
    - order: Filter order (default: 4).
    1
    Returns:
    - Filtered signal (1D array).
    """
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist
    b, a = butter(order, normalized_cutoff, btype='high', analog=False)
    return lfilter(b, a, data)

# Compute zero-sequence voltage (U0) and current (I0)
def compute_zero_sequence(voltages, currents):
    """
    Computes the zero-sequence voltage and current (U0 and I0).
    
    Parameters:
    - voltages: List of 3-phase voltage signals (list of arrays).
    - currents: List of 3-phase current signals (list of arrays).
    
    Returns:
    - u0: Zero-sequence voltage (1D array).
    - i0: Zero-sequence current (1D array).
    """
    # Zero-sequence components are the mean of the 3-phase signals
    u0 = np.mean(voltages, axis=0)
    i0 = np.mean(currents, axis=0)
    return u0, i0

# Calculate active power (W) using U0, I0, and cos(φ)
def calculate_active_power(u0, i0):
    """
    Calculates active power W based on U0, I0, and cos(φ).
    
    Parameters:
    - u0: Zero-sequence voltage (1D array).
    - i0: Zero-sequence current (1D array).
    
    Returns:
    - Active power W (1D array).
    """
    cos_phi = np.cos(np.angle(u0) - np.angle(i0))  # Phase relationship
    w = np.real(u0 * np.conj(i0)) * cos_phi  # Active power calculation
    return w

# Classify the fault based on W and threshold E
def classify_wattmetric_fault(w, e_threshold):
    """
    Classifies a fault as 'Forward Fault' or 'Reverse Fault' using active power W.
    
    Parameters:
    - w: Active power W (1D array).
    - e_threshold: Threshold value for fault classification.
    
    Returns:
    - String indicating fault type ('Forward Fault' or 'Reverse Fault').
    """
    mean_w = np.mean(w)  # Average active power
    if mean_w > e_threshold:
        return "Reverse Fault"
    else:
        return "Forward Fault"
    
def compute_threshold(w, method="scaled", k=3, factor=10):
    """
    Computes the threshold for fault classification.

    Parameters:
    - w: Active power W (1D array).
    - method: Method to compute the threshold ("scaled" or "statistical").
    - k: Scaling factor for statistical threshold (default: 3).
    - factor: Multiplier for scaled threshold (default: 10).

    Returns:
    - Computed threshold E.
    """
    if method == "statistical":
        # Mean + k * standard deviation
        mu = np.mean(w)
        sigma = np.std(w)
        e_threshold = mu + k * sigma
    elif method == "scaled":
        # Use mean of active power scaled by a factor
        e_threshold = np.mean(w) * factor
    else:
        raise ValueError("Unsupported method for threshold calculation.")
    
    return e_threshold


def detect_fault_with_timestamp(w, timestamps, threshold):
    """
    Detects the first fault condition based on active power and returns the timestamp.

    Parameters:
    - w: Active power W (1D array).
    - timestamps: Array of timestamps corresponding to W.
    - threshold: Threshold value for fault detection.

    Returns:
    - Tuple (fault_time, fault_type), where fault_time is the timestamp of the first detected fault
      and fault_type is 'Forward Fault' or 'Reverse Fault'.
    """
    # Find the indices where W crosses the threshold
    fault_indices = np.where(w > threshold)[0]
    if len(fault_indices) > 0:
        fault_time = timestamps[fault_indices[0]]  # First occurrence
        return fault_time
    else:
        return None  # No fault detected

def process_wattmetric_method_with_timestamp(folder_path, cutoff_freq=50.0, e_method="scaled", k=3, factor=10):
    """
    Processes Comtrade files and applies the Wattmetric Method for fault detection with timestamps.
    
    Parameters:
    - folder_path: Path to the folder containing Comtrade files (.cfg and .dat).
    - cutoff_freq: High-pass filter cutoff frequency (default: 50 Hz).
    """
    files = [f for f in os.listdir(folder_path) if f.endswith('.cfg')]
    for cfg_file in files:
        dat_file = cfg_file.replace('.cfg', '.dat')

        if not os.path.exists(os.path.join(folder_path, dat_file)):
            print(f"Dat file not found for {cfg_file}. Skipping.")
            continue

        # Load the Comtrade data
        comtradeObj = pyComtrade.ComtradeRecord()
        comtradeObj.read(os.path.join(folder_path, cfg_file), os.path.join(folder_path, dat_file))
        
        # Extract time, voltage, and current data
        timestamps = comtradeObj.get_timestamps()
        voltages = [ch['values'] for ch in comtradeObj.cfg_data['A'][:3]]  # 3-phase voltages
        currents = [ch['values'] for ch in comtradeObj.cfg_data['A'][3:]]  # 3-phase currents

        # Ensure dimensions match timestamps
        min_length = len(timestamps)
        voltages = [v[:min_length] for v in voltages]
        currents = [c[:min_length] for c in currents]

        # High-pass filter the signals
        sampling_rate = 1 / np.mean(np.diff(timestamps))  # Sampling frequency
        filtered_voltages = [high_pass_filter(v, cutoff_freq, sampling_rate) for v in voltages]
        filtered_currents = [high_pass_filter(i, cutoff_freq, sampling_rate) for i in currents]

        # Compute zero-sequence components
        u0, i0 = compute_zero_sequence(filtered_voltages, filtered_currents)

        # Calculate active power
        w = calculate_active_power(u0, i0)

        # Compute dynamic threshold
        e_threshold = compute_threshold(w, method=e_method, k=k, factor=factor)
        print(f"Dynamic Threshold E: {e_threshold}")

        # Detect fault with timestamp
        fault_time = detect_fault_with_timestamp(w, timestamps, e_threshold)
        if fault_time is not None:
            # Classify fault
            fault_type = classify_wattmetric_fault(w, e_threshold)
            print(f"File: {cfg_file}, Fault Type: {fault_type}, Fault Time: {fault_time:.6f} seconds")
        else:
            print(f"File: {cfg_file}, No Fault Detected")

# Path to the folder containing Comtrade files
folder_path = "comdata"
process_wattmetric_method_with_timestamp(folder_path)
