import os
import numpy as np
import matplotlib.pyplot as plt
from utilities.signal_processing import high_pass_filter
from utilities.fault_classification import detect_fault_with_thresholds, classify_fault
from utilities.threshold_compute import compute_thresholds
from utilities import pyComtrade


# Compute zero-sequence voltage (U0) and current (I0)
def compute_zero_sequence(voltages, currents, timestamps):
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


# Classify the fault based on Power
def classify_fault(u0, i0, power_threshold, file_name, power_data=None):
    """
    Classifies a fault as 'Forward Fault' or 'Reverse Fault' based on power and phase difference.
    Dynamically adjusts power threshold based on characteristics of the power signal.

    Parameters:
    - u0: Zero-sequence voltage (1D array).
    - i0: Zero-sequence current (1D array).
    - power_threshold: Power threshold for classification.
    - file_name: The name of the file (e.g., "rspe28", "rspe29").
    - power_data: Calculated power data (optional).

    Returns:
    - String indicating fault type ('Forward Fault', 'Reverse Fault', or 'Deadzone!!').
    """
    # If power_data is provided, use it, else calculate power as usual
    if power_data is None:
        phase_angle = np.angle(u0) - np.angle(i0)
        cos_phi = np.cos(phase_angle)
        power_data = np.abs(u0) * np.abs(i0) * cos_phi  # Active power

    print("First 20 Power values (W):", power_data[:20])  # Print first 20 power values for inspection
    print("Power threshold:", power_threshold)

    # Dynamically adjust power threshold based on statistical properties (std deviation or mean)
    # Use a simpler dynamic adjustment or apply a fixed multiplier
    power_max = np.max(np.abs(power_data))
    adjusted_threshold = power_max * 0.1  # Example: 10% of max power as threshold for fault detection
    print(f"Adjusted power threshold for {file_name}: {adjusted_threshold:.6e}")

    # Check for first occurrence where power crosses the threshold
    fault_type = "Deadzone!!"  # Default state
    for i, power in enumerate(power_data):
        if power > adjusted_threshold:  # If power exceeds the threshold in positive direction
            print(f"At index {i}, power {power} exceeds threshold. Reverse Fault Detected.")
            fault_type = "Reverse Fault Detected"
            break
        elif power < -adjusted_threshold:  # If power exceeds the negative threshold
            print(f"At index {i}, power {power} is below negative threshold. Forward Fault Detected.")
            fault_type = "Forward Fault Detected"
            break

    return fault_type



def compute_power_threshold(filtered_voltages, filtered_zero_seq_current, timestamps,
                                           fault_free_start=0.0, fault_free_end=0.2):
    """
    Compute the power threshold for the wattmetric method using fault-free data.

    Args:
        filtered_voltages: List of filtered voltage signals.
        filtered_zero_seq_current: Filtered zero-sequence current signal.
        timestamps: Array of timestamps corresponding to the data.
        fault_free_start: Start time of the fault-free region (default: 0.0).
        fault_free_end: End time of the fault-free region (default: 0.2).

    Returns:
        float: Calculated power threshold.
    """
    # Identify the fault-free region based on timestamps
    fault_free_indices = (timestamps >= fault_free_start) & (timestamps <= fault_free_end)

    # Extract fault-free data for zero-sequence voltage (U0) and current (I0)
    fault_free_voltages = [v[fault_free_indices] for v in filtered_voltages]
    fault_free_current = filtered_zero_seq_current[fault_free_indices]

    # Compute U0 as the mean of the three-phase voltages in the fault-free region
    u0 = np.mean(fault_free_voltages, axis=0)

    # Compute the phase angle between U0 and I0
    phase_angle = np.angle(u0) - np.angle(fault_free_current)
    cos_phi = np.cos(phase_angle)

    # Compute active power (W) in the fault-free region
    power_fault_free_region = np.abs(u0) * np.abs(fault_free_current) * cos_phi

    # Compute the mean and standard deviation of the active power in the fault-free region
    mean_power = np.mean(power_fault_free_region)
    std_power = np.std(power_fault_free_region)

    # Return computed threshold based on the mean and standard deviation
    threshold = mean_power + 3 * std_power  # Using n=3 for sensitivity

    print(f"Power Threshold: {threshold:.6e}")

    return threshold

def process_comtrade_data(folder_path, cutoff_freq=1):
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
        # Debugging: Check if i0 has any values
        print("Zero-sequence current (i0) before filtering:", raw_zero_seq_current[:10])  # Print first 10 samples

        # Ensure dimensions match timestamps
        min_length = len(timestamps)
        voltages = [v[:min_length] for v in raw_voltages]
        if raw_zero_seq_current is not None:
            zero_seq_current = raw_zero_seq_current[:min_length]

        # High-pass filter the signals
        sampling_rate = 10e3  # Sampling frequency
        filtered_voltages = [high_pass_filter(v, cutoff_freq, sampling_rate) for v in voltages]
        filtered_zero_seq_current = high_pass_filter(zero_seq_current, cutoff_freq, sampling_rate)

        print("Zero-sequence current (i0) after filtering:", filtered_zero_seq_current[:10])  # First 10 samples

        # Compute zero-sequence voltage (U0)
        u0 = np.mean(filtered_voltages, axis=0)

        # Plot U0 and I0 signals
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, np.abs(u0), label="Zero-sequence Voltage (|U0|)")
        plt.plot(timestamps, np.abs(filtered_zero_seq_current), label="Zero-sequence Current (|I0|)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.title(f"Zero-sequence Voltage and Current {cfg_file}")
        plt.grid()
        plt.show()

        # Compute thresholds
        u0_threshold, i0_threshold = compute_thresholds(comtradeObj.cfg_data, filtered_zero_seq_current)
        power_threshold = compute_power_threshold(filtered_voltages, filtered_zero_seq_current, timestamps)



        # Detect faults
        result = detect_fault_with_thresholds(u0, filtered_zero_seq_current, timestamps, u0_threshold, i0_threshold)
        if result[0]:
            # Classify the fault
            fault_type = classify_fault(u0, filtered_zero_seq_current, power_threshold, cfg_file)
            print(f"Fault detected in {cfg_file} at {result[1]:.6f} seconds. Fault type: {fault_type} .")
        else:
            print(f"No fault detected in {cfg_file}.")


folder_path = "comdata"
process_comtrade_data(folder_path)




