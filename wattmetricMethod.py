import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import pyComtrade
from utilities.signal_processing import high_pass_filter
from utilities.threshold_compute import compute_thresholds
from utilities.fault_classification import detect_fault_with_thresholds

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

        # Extract raw voltages, zero-sequence current, and currents
        raw_voltages = [ch['values'] for ch in comtradeObj.cfg_data['A'][0:3]]  # Channels 1, 2, 3
        raw_zero_seq_current = comtradeObj.cfg_data['A'][3]['values']  # Channel 4
        # raw_currents = [ch['values'] for ch in comtradeObj.cfg_data['A'][4:7]]  # Channels 5, 6, 7

        # Ensure dimensions match timestamps
        min_length = len(timestamps)
        voltages = [v[:min_length] for v in raw_voltages]
        # currents2 = [c[:min_length] for c in currents]
        if raw_zero_seq_current is not None:
            zero_seq_current = raw_zero_seq_current[:min_length]


        # High-pass filter the signals
        sampling_rate = 10e3  # Sampling frequency
        filtered_voltages = [high_pass_filter(v, cutoff_freq, sampling_rate) for v in voltages]
        filtered_zero_seq_current = high_pass_filter(zero_seq_current, cutoff_freq, sampling_rate)

        # Compute zero-sequence voltage (U0)
        u0 = np.mean(filtered_voltages, axis=0)

        # Calculate active power
        w = calculate_active_power(u0, filtered_zero_seq_current)

        # Compute dynamic threshold
        u0_threshold, i0_threshold = compute_thresholds(comtradeObj.cfg_data, filtered_zero_seq_current)
        result = detect_fault_with_thresholds(u0, filtered_zero_seq_current, timestamps, u0_threshold, i0_threshold)

        # Detect fault with timestamp
        fault_time = detect_fault_with_timestamp(w, timestamps, 0)
        if result[0]:
            # e_threshold = compute_threshold_direction(w, method=e_method, k=k, factor=factor)
            print(f"Dynamic Threshold E: {0}")
            # Classify fault
            fault_type = classify_wattmetric_fault(w, 0)
            print(f"File: {cfg_file}, Fault Type: {fault_type}, Fault Time: {fault_time:.6f} seconds")
        else:
            print(f"File: {cfg_file}, No Fault Detected")

# Path to the folder containing Comtrade files
folder_path = "comdata"
process_wattmetric_method_with_timestamp(folder_path)
