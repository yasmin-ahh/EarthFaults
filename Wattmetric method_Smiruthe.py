import os
import numpy as np
import matplotlib.pyplot as plt
from utilities.fault_classification import detect_fault_with_thresholds_wattmetric
from utilities.threshold_compute import compute_thresholds_wattmetric
from utilities import pyComtrade

def classify_wattmetric(u0, i0, timestamps, fault_time):
    """
    Classifies the fault direction using the Wattmetric method after a fault is detected.

    Parameters:
    - u0: Zero-sequence voltage (1D array, complex values).
    - i0: Zero-sequence current (1D array, complex values).
    - timestamps: Array of timestamps corresponding to the signals.
    - start_time: Time when the fault was detected.
    - threshold: Threshold for active power to classify direction (default: 0).

    Returns:
    - fault_direction: "Forward Fault" or "Reverse Fault".
    """
    # Find the starting index for the 3-second window
    start_index = np.searchsorted(timestamps, fault_time)
    start_time = timestamps[start_index ] + 0.2

    # Define the end of the 3-second window
    end_time = start_time + 3.0
    end_index = np.searchsorted(timestamps, end_time)

    # Extract data within the 3-second window
    u0_window = u0[start_index:end_index]
    i0_window = i0[start_index:end_index]

    # Calculate magnitudes and phase difference
    u0_magnitude = np.abs(u0_window)
    i0_magnitude = np.abs(i0_window)
    phase_diff = np.angle(u0_window) - np.angle(i0_window)
    phi_threshold = 0.2 * np.mean(phase_diff)
    active_power = u0_magnitude * i0_magnitude * np.cos(phase_diff)
    adjusted_threshold = 2

    fault_type = "Deadzone!!"  # Default state
    for i, power in enumerate(active_power):
        if (phase_diff > phi_threshold).any() and (power > adjusted_threshold).any():  # If power exceeds the threshold in positive direction
            fault_type = "Reverse Fault Detected"
            break
        elif (phase_diff > phi_threshold).any() and (power < -adjusted_threshold).any():  # If power exceeds the negative threshold
            fault_type = "Forward Fault Detected"
            break

    return fault_type

def process_comtrade_data(folder_path):
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

        # Ensure dimensions match timestamps
        min_length = len(timestamps)
        voltages = [v[:min_length] for v in raw_voltages]
        if raw_zero_seq_current is not None:
            zero_seq_current = raw_zero_seq_current[:min_length]
        # Compute zero-sequence voltage (U0)
        u0 = np.mean(voltages, axis=0)

        # Plot U0 and I0 signals
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, np.abs(u0), label="Zero-sequence Voltage (|U0|)")
        plt.plot(timestamps, np.abs(zero_seq_current), label="Zero-sequence Current (|I0|)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.title(f"Zero-sequence Voltage and Current {cfg_file}")
        plt.grid()
        plt.show()

        # Compute thresholds
        u0_threshold = compute_thresholds_wattmetric(comtradeObj.cfg_data)

        # Detect faults
        result = detect_fault_with_thresholds_wattmetric(u0, timestamps, u0_threshold)
        if result[0]:
            fault_time = result[1][0]

            # Classify the fault
            fault_type = classify_wattmetric(u0, zero_seq_current, timestamps, fault_time)
            print(f"Fault detected in {cfg_file} at {result[1][0]:} seconds. Fault type: {fault_type} .")
        else:
            print(f"No fault detected in {cfg_file}.")


folder_path = "comdata"
process_comtrade_data(folder_path)
