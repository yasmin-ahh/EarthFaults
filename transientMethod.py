__version__ = "$Revision$"  # SVN revision.
__date__ = "$Date$"         # Date of the last SVN revision.

import os
import numpy as np
import matplotlib.pyplot as plt
import pyComtrade
from utilities.signal_processing import high_pass_filter

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

# Classify the fault based on the phase difference between U0 and I0
def classify_fault(u0, i0):
    """
    Classifies a fault as 'Forward Fault' or 'Reverse Fault' based on phase difference.
    
    Parameters:
    - u0: Zero-sequence voltage (1D array).
    - i0: Zero-sequence current (1D array).
    
    Returns:
    - String indicating fault type ('Forward Fault' or 'Reverse Fault').
    """
    # Calculate the phase difference
    phase_difference = np.mean(np.angle(np.exp(1j * (np.angle(u0) - np.angle(i0)))))

    # Classify based on phase difference
    if phase_difference > 0:
        return "Forward Fault"
    else:
        return "Reverse Fault"


def detect_single_transient(filtered_voltages, timestamps, threshold):
    """
    Detects a single transient in the filtered voltages and returns the fault timestamp.

    Parameters:
    - filtered_voltages: List of filtered voltage signals (list of arrays).
    - timestamps: Array of timestamps corresponding to the signals.
    - threshold: Threshold value for transient detection.

    Returns:
    - Tuple (fault_phase, fault_time), where fault_phase is the phase number (1-based),
      and fault_time is the timestamp of the first detected fault.
      Returns None if no fault is detected.
    """
    for phase, voltage in enumerate(filtered_voltages):
        # Identify points where the absolute value exceeds the threshold
        fault_indices = np.where(np.abs(voltage) > threshold)[0]
        print(f"Transient Indices for Phase {phase + 1}: {fault_indices}")  # Debugging

        if len(fault_indices) > 0:
            fault_index = fault_indices[0]  # First occurrence
            fault_time = timestamps[fault_index]
            print(f"Fault Time for Phase {phase + 1}: {fault_time}")  # Debugging
            return (phase + 1, fault_time)

    return None  # No fault detected


def process_comtrade_data(folder_path, cutoff_freq=50.0):
    """
    Processes Comtrade files in a folder and identifies transient faults.

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

        # Extract voltages (Channels 2, 3, 4 -> Python indexing [1:4])
        voltages = [ch['values'] for ch in comtradeObj.cfg_data['A'][0:3]]  # 3-phase voltages (L1, L2, L3)

        # Check for zero-sequence current explicitly
        zero_seq_current = None
        if len(comtradeObj.cfg_data['A']) > 3:
            zero_seq_current = comtradeObj.cfg_data['A'][3]['values']  # Explicit I_3I0 if available
            print(f"Zero-sequence current (I_3I0) provided in {cfg_file}. Using the explicit measurement.")


        # Extract currents (Channels 5, 6, 7 -> Python indexing [3:6])
        currents = [ch['values'] for ch in comtradeObj.cfg_data['A'][4:7]]  # 3-phase currents (I L1, I L2, I L3)

        # Ensure dimensions match timestamps
        min_length = len(timestamps)
        voltages = [v[:min_length] for v in voltages]
        currents = [c[:min_length] for c in currents]

        # Plot raw voltage and current signals
        plt.figure(figsize=(12, 6))
        for i in range(3):  # Loop over the three phases (indices 0, 1, 2)
            plt.plot(timestamps, voltages[i], label=f'Voltage Phase {i + 1}')
            plt.plot(timestamps, currents[i], label=f'Current Phase {i + 1}')
        plt.title(f'Raw Signals from {cfg_file}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid()
        plt.show()

        # High-pass filter the signals
        sampling_rate = 1 / np.mean(np.diff(timestamps))  # Sampling frequency
        filtered_voltages = [high_pass_filter(v, cutoff_freq, sampling_rate) for v in voltages]
        filtered_currents = [high_pass_filter(i, cutoff_freq, sampling_rate) for i in currents]

        # Compute zero-sequence components only if I_3I0 is not provided
        if zero_seq_current is None:
            print(f"No explicit zero-sequence current (I_3I0) found. Calculating it from the 3-phase currents.")
            u0, i0 = compute_zero_sequence(filtered_voltages, filtered_currents)
        else:
            print(f"Using explicit zero-sequence current for fault classification.")
            u0 = np.mean(filtered_voltages, axis=0)  # Zero-sequence voltage
            i0 = zero_seq_current  # Use the explicit I_3I0

        # Detect transient conditions and find the timestamp
        threshold = np.mean([np.mean(np.abs(v)) for v in filtered_voltages]) + 3 * np.std([np.std(np.abs(v)) for v in filtered_voltages])
        print(f"Threshold for transient detection: {threshold}")  # Debugging
        
        transient = detect_single_transient(filtered_voltages, timestamps, threshold)
        if transient:
            fault_phase, fault_time = transient
            print(f"Transient detected in {cfg_file}.")
            print(f"Fault detected in Phase {fault_phase} at {fault_time:.6f} seconds.")


            # Classify the fault
            fault_type = classify_fault(u0, i0)
            print(f"Fault type: {fault_type}")
        else:
            print(f"No transient detected in {cfg_file}.")



# Path to the folder containing Comtrade files
folder_path = "comdata"
process_comtrade_data(folder_path)
