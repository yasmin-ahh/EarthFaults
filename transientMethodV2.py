import os
import numpy as np
import matplotlib.pyplot as plt
from utilities.signal_processing import high_pass_filter
from utilities.fault_classification import detect_fault_with_thresholds, classify_fault
from utilities.threshold_compute import compute_thresholds
import pyComtrade

def scale_data(raw_data, a, b):
    """
    Scales raw data using the given scaling factor 'a' and offset 'b'.

    Parameters:
    - raw_data: Raw signal values (1D array).
    - a: Scaling factor.
    - b: Offset.

    Returns:
    - Scaled data as a 1D array.
    """
    return raw_data * a + b

def scale_all_channels(raw_signals, cfg_channels):
    """
    Scales all channels using their respective scaling factors (a and b) from the cfg file.

    Parameters:
    - raw_signals: List of raw signal arrays (1D lists or arrays).
    - cfg_channels: Configuration data for the channels, containing 'a' and 'b'.

    Returns:
    - List of scaled signals (NumPy arrays).
    """
    scaled_signals = []
    for i in range(len(raw_signals)):
        scaled_signals.append(scale_data(np.array(raw_signals[i]), cfg_channels[i]['a'], cfg_channels[i]['b']))

    return scaled_signals


def process_comtrade_data(folder_path, cutoff_freq=50.0):
    """
    Processes Comtrade files in a folder and identifies faults based on U0 and I0 thresholds.

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

        # Compute thresholds
        u0_threshold, i0_threshold = compute_thresholds(comtradeObj.cfg_data, filtered_zero_seq_current)

        # Detect faults
        result = detect_fault_with_thresholds(u0, filtered_zero_seq_current, timestamps, u0_threshold, i0_threshold)
        if result[0]:
            # Classify the fault
            fault_type = classify_fault(u0, filtered_zero_seq_current)
            print(f"Fault detected in {cfg_file} at {result[1]:.6f} seconds. Fault type: {fault_type} .")
        else:
            print(f"No fault detected in {cfg_file}.")


# Path to the folder containing Comtrade files
folder_path = "comdata"
process_comtrade_data(folder_path)
