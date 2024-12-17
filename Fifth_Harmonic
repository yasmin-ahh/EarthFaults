# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 19:23:42 2024

@author: PSP19
"""

import os
import numpy as np
import matplotlib.pyplot as plt
#from utilities.signal_processing import high_pass_filter
#from utilities.fault_classification import detect_fault_with_thresholds, classify_fault
#from utilities.threshold_compute import compute_thresholds
from utilities import pyComtrade


def process_comtrade_data(folder_path):
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
        #comtradeObj.read(os.path.join(folder_path,"rspe4.cfg"), os.path.join(folder_path, "rspe4.dat"))
        # Extract time, voltage, and current data
        timestamps = comtradeObj.get_timestamps()

        # Extract raw voltages, zero-sequence current, and currents
        raw_voltages = [ch['values'] for ch in comtradeObj.cfg_data['A'][0:3]]  # Channels 1, 2, 3
        raw_zero_seq_current = comtradeObj.cfg_data['A'][3]['values']  # Channel 4
        raw_currents = [ch['values'] for ch in comtradeObj.cfg_data['A'][4:7]]  # Channels 5, 6, 7

        # Ensure dimensions match timestamps
        min_length = len(timestamps)
        voltages = [v[:min_length] for v in raw_voltages]
        currents = [c[:min_length] for c in raw_currents]
        if raw_zero_seq_current is not None:
            zero_seq_current = raw_zero_seq_current[:min_length]
        
        # Subplot for voltages
        plt.subplot(2, 1, 1)
        plt.plot(timestamps, voltages[0], label='Ua', color='blue')
        plt.plot(timestamps, voltages[1], label='Ub', color='red')
        plt.plot(timestamps, voltages[2], label='Uc', color='green')
        plt.title("Three-Phase Voltages")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.legend()
        plt.grid()

        # Subplot for currents
        plt.subplot(2, 1, 2)
        plt.plot(timestamps, currents[0], label='Ia', color='red')
        plt.plot(timestamps, currents[1], label='Ib', color='blue')
        plt.plot(timestamps, currents[2], label='Ic', color='green')
        plt.title("Three-Phase Currents")
        plt.xlabel("Time (s)")
        plt.ylabel("Current (A)")
        plt.legend()
        plt.grid()    
      
        return voltages,currents,zero_seq_current, timestamps

"""            
#get input  data from Comtrade files
def getInputData():
    # File paths (replace with your actual file paths)
    cfg_file = r"D:\Study Material\Project\Data\rspe4.cfg"
    dat_file = r"D:\Study Material\Project\Data\rspe4.dat"
    
    # Step 1: Parse the .cfg file
    cfg_metadata = {}
    analog_channel_labels = []
    digital_channel_labels = []
    
    with open(cfg_file, "r") as cfg:
        lines = cfg.readlines()
    
        # Extract metadata
        cfg_metadata['station_name'], cfg_metadata['rec_dev_id'], cfg_metadata['rev_year'] = lines[0].strip().split(",")
        second_line = lines[1].strip().split(",")
        analog_count = int(''.join(filter(str.isdigit, second_line[1])))
        digital_count = int(''.join(filter(str.isdigit, second_line[2])))
    
        # Extract analog channel labels
        for i in range(2, 2 + analog_count):
            parts = lines[i].strip().split(",")
            analog_channel_labels.append(parts[1])  # Channel name
    
    # Step 2: Parse the .dat file
    timestamps = []
    analog_samples = []
    digital_samples = []
    
    with open(dat_file, "r") as dat:
        for line in dat:
            parts = line.strip().split(",")
            if len(parts) >= 2 + analog_count + digital_count:
                try:
                    timestamps.append(float(parts[1]))
                    analog_samples.append([float(x) for x in parts[2:2 + analog_count]])
                    digital_samples.append([int(x) for x in parts[2 + analog_count:]])
                except ValueError:
                    pass
    
    # Convert lists to numpy arrays
    timestamps = np.array(timestamps)
    analog_samples = np.array(analog_samples)
    digital_samples = np.array(digital_samples)
    
    # Split voltage and current signals
    voltages = analog_samples[:, :3]  # Columns for Ua, Ub, Uc
    currents = analog_samples[:, 4:7]  # Columns for Ia, Ib, Ic
    
    return voltages, currents,timestamps
"""
# Compute Zero-Sequence Components
def compute_zero_sequence(voltages, currents):
    """
    Compute zero-sequence voltage and current components.
    Args:
        voltages (array): Voltage samples [Ua, Ub, Uc].
        currents (array): Current samples [Ia, Ib, Ic].
    Returns:
        tuple: Zero-sequence voltage (U0) and current (I0).
    """
    U0 = np.mean(voltages, axis=1)  # Zero-sequence voltage
    I0 = np.mean(currents, axis=1)  # Zero-sequence current
    return U0, I0

def extract_fifth_harmonic(signal, sampling_rate, fundamental_frequency):
    """
    Extract the 5th harmonic of a signal using FFT.

    Args:
        signal (array): The input time-domain signal (voltages or currents).
        sampling_rate (float): Sampling rate of the signal in Hz.
        fundamental_frequency (float): Fundamental frequency in Hz.

    Returns:
        array: Time-domain signal corresponding to the 5th harmonic.
    """
    # Total number of samples
    N = len(signal)

    # Perform FFT on the signal
    fft_result = np.fft.fft(signal)

    # Frequency resolution (step size between frequencies)
    freq_resolution = sampling_rate / N

    # Calculate the index for the 5th harmonic
    fifth_harmonic_freq = 5 * fundamental_frequency  # Frequency of the 5th harmonic
    index = int(np.round(fifth_harmonic_freq / freq_resolution))

    # Create an empty FFT spectrum with zeros
    fifth_harmonic_spectrum = np.zeros_like(fft_result, dtype=complex)

    # Isolate the 5th harmonic and its conjugate (negative frequency part)
    fifth_harmonic_spectrum[index] = fft_result[index]  # Positive frequency
    fifth_harmonic_spectrum[-index] = fft_result[-index]  # Negative frequency (conjugate)

    # Perform the Inverse FFT to get the time-domain signal of the 5th harmonic
    fifth_harmonic_signal = np.fft.ifft(fifth_harmonic_spectrum).real

    return fifth_harmonic_signal


def extract_fifth_harmonics(voltages, currents, sampling_rate, fundamental_frequency):
    """
    Extract the 5th harmonic components for voltage and current arrays.

    Args:
        voltages (list of arrays): List of voltage signals (e.g., Phase A, B, C).
        currents (list of arrays): List of current signals (e.g., Phase A, B, C).
        sampling_rate (float): Sampling rate of the signals in Hz.
        fundamental_frequency (float): Fundamental frequency in Hz.

    Returns:
        dict: Dictionary containing 5th harmonic components for voltages and currents.
    """
    # Extract 5th harmonics for each voltage and current phase
    fifth_harmonics_voltages = [
        extract_fifth_harmonic(v, sampling_rate, fundamental_frequency) for v in voltages
    ]

    fifth_harmonics_currents = [
        extract_fifth_harmonic(c, sampling_rate, fundamental_frequency) for c in currents
    ]

    return fifth_harmonics_voltages,fifth_harmonics_currents
    

# Fault Detection Logic
def detect_fault(U0, I0, threshold):
    """
    Detect faults based on zero-sequence components and active power.
    Args:
        U0 (float): Zero-sequence voltage.
        I0 (float): Zero-sequence current.
        threshold (float): Fault threshold.
    Returns:
        str: Fault detection result.
    """
    phase_angle = np.angle(U0) - np.angle(I0)  # Phase angle of zero-sequence components
    sin_phi = np.sin(phase_angle)
    W = U0 * I0 * sin_phi  # Reactive power
    fault_time = None
    
    # Check for the first instance where rective power crosses the threshold
    for i, power in enumerate(W):
        if power > threshold:
            results = "Forward Fault Detected"
            fault_time = timestamps[i]
            break
        elif power < -threshold:
            results = "Reverse Fault Detected"
            fault_time = timestamps[i]
            break
    print("\nReactive Power : ", np.mean(W))
    print(f"Fault Time: {fault_time if fault_time else 'No Fault Detected'} ")
    print("Fault type : ", results)
    return results, fault_time

# Calculating Threshold from Fault-free data
def compute_threshold(U0, I0, fault_free_start, fault_free_end, timestamps, n=3):
    """
    Compute threshold for fault detection based on fault-free data.

    Args:
        U0 (array): Zero-sequence voltage array.
        I0 (array): Zero-sequence current array.
        fault_free_start (float): Start time of fault-free region (seconds).
        fault_free_end (float): End time of fault-free region (seconds).
        timestamps (array): Array of timestamps corresponding to U0 and I0.
        n (int): Sensitivity multiplier for threshold calculation.

    Returns:
        float: Calculated threshold.
    """
    # Identify fault-free data indices based on the given time range
    fault_free_indices = (timestamps >= fault_free_start) & (timestamps <= fault_free_end)

    # Extract fault-free data
    fault_free_U0 = U0[fault_free_indices]
    fault_free_I0 = I0[fault_free_indices]

    # Compute reactive power (W) in fault-free region
    phase_angle = np.angle(fault_free_U0) - np.angle(fault_free_I0)
    sin_phi = np.sin(phase_angle)
    fault_free_W = fault_free_U0 * fault_free_I0 * sin_phi

    # Compute mean and standard deviation of reactive power
    mean_W = np.mean(fault_free_W)
    std_W = np.std(fault_free_W)

    # Calculate threshold using mean + n * std deviation
    threshold = mean_W + n * std_W
    print(f"Computed Threshold: {threshold}")

    return threshold

# Path to the folder containing Comtrade files
folder_path = "comdata"
voltages,currents,zero_seq,timestamps = process_comtrade_data(folder_path)

fifth_Harmonic_voltages = []
fifth_Harmonic_currents = []
zero_seq_I = []
zero_seq_V = []
fundamental_frequency =50
sampling_frequency = 100000


fifth_Harmonic_voltages,fifth_Harmonic_currents = extract_fifth_harmonics(voltages, currents, sampling_frequency, fundamental_frequency)
time = np.arange(0, 1, 1 / sampling_frequency)  # 1-second signal
'''
# Plot the extracted 5th harmonic voltages and currents
fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)

for i in range(3):
    # Plot extracted 5th harmonic for voltage
    axes[i, 0].plot(timestamps[i], fifth_Harmonic_voltages[i], label=f"Phase {['A', 'B', 'C'][i]} Voltage Harmonic", color='tab:blue')
    axes[i, 0].set_ylabel("Amplitude")
    axes[i, 0].legend(loc='upper right')
    
    # Plot extracted 5th harmonic for current
    axes[i, 1].plot(timestamps[i], fifth_Harmonic_currents[i], label=f"Phase {['A', 'B', 'C'][i]} Current Harmonic", color='tab:orange')
    axes[i, 1].set_xlabel("Time [s]")
    axes[i, 1].set_ylabel("Amplitude")
    axes[i, 1].legend(loc='upper right')

'''
# Compute threshold
# Define fault-free time range (e.g., 0 to 2 seconds)
fault_free_start = 0  # Start time in seconds
fault_free_end = 1000  # End time in seconds

THRESHOLD =1
# Detect faults
fault_results = detect_fault(fifth_Harmonic_voltages, fifth_Harmonic_currents, THRESHOLD)


   
