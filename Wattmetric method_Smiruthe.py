import os
import numpy as np
import matplotlib.pyplot as plt
from utilities.fault_classification import detect_fault_with_thresholds
from utilities.threshold_compute import compute_thresholds
from utilities import pyComtrade


# Classify the fault based on Power
def classify_fault_wattmetric(u0, i0, timestamps,cfg_file, power_data=None):

    if power_data is None:
        phase_angle = np.angle(u0) - np.angle(i0)
        cos_phi = np.cos(phase_angle)
        power_data = np.abs(u0) * np.abs(i0) * cos_phi  # Active power

    # Dynamically adjust power threshold based on statistical properties (std deviation or mean)
    # power_max = np.max(np.abs(power_data))
    adjusted_threshold = 20000


    # Plot Active power vs Time
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, power_data, label=" Active Power (W)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title(f"Active Power {cfg_file}")
    plt.grid()
    plt.show()

    # Check for first occurrence where power crosses the threshold
    fault_type = "Deadzone!!"  # Default state
    for i, power in enumerate(power_data):
        if power > adjusted_threshold:  # If power exceeds the threshold in positive direction
            fault_type = "Reverse Fault Detected"
            break
        elif power < -adjusted_threshold:  # If power exceeds the negative threshold
            fault_type = "Forward Fault Detected"
            break

    return fault_type

def process_comtrade_data(folder_path, cutoff_freq=10):
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

        # High-pass filter the signals
        sampling_rate = 10e3  # Sampling frequency
        filtered_voltages = voltages
        filtered_zero_seq_current = zero_seq_current

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

        # Detect faults
        result = detect_fault_with_thresholds(u0, filtered_zero_seq_current, timestamps, u0_threshold, i0_threshold)
        if result[0]:
            # Classify the fault
            fault_type = classify_fault_wattmetric(u0, filtered_zero_seq_current, timestamps , cfg_file)
            print(f"Fault detected in {cfg_file} at {result[1]:.6f} seconds. Fault type: {fault_type} .")
        else:
            print(f"No fault detected in {cfg_file}.")


folder_path = "comdata"
process_comtrade_data(folder_path)


