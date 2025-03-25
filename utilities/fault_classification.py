import numpy as np
import matplotlib.pyplot as plt
import time

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
        return "Forward"
    else:
        return "Reverse"

def classify_fault_wattmetric(u0, i0):
    cos_phi = np.cos(np.angle(u0) - np.angle(i0))  # Phase relationship
    w = np.real(u0 * np.conj(i0)) * cos_phi  # Active power calculation


def classify_fault_wattmetric(u0, i0, timestamps, start_time, threshold=0):
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
    start_index = np.searchsorted(timestamps, start_time)

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
    # print("Phase Difference: ", phase_diff)

    # Calculate active power explicitly
    active_power = np.mean(u0_magnitude * i0_magnitude * np.cos(phase_diff))
    # print("Active Power: ", active_power)

    # Compare active power to the threshold
    if active_power < 0:
        fault_direction = "Forward Fault"
    else:
        fault_direction = "Reverse Fault"

    return fault_direction

def detect_fault_with_thresholds_wattmetric(u0, timestamps, u0_threshold):
    """
    Detects a fault based on zero-sequence voltage and current thresholds.

    Parameters:
    - u0: Zero-sequence voltage (1D array).
    - i0: Zero-sequence current (1D array).
    - timestamps: Array of timestamps corresponding to the signals.
    - u0_threshold: Threshold for zero-sequence voltage.
    - i0_threshold: Threshold for zero-sequence current.

    Returns:
    - fault_time: Timestamp when the fault is first detected, or None if no fault.
    """
    fault_detected = False
    fault_indices = []

    # Iterate through the length of u0 (assuming u0, i0, and timestamps are of the same length)
    fault_indices = np.where(np.abs(u0) > u0_threshold)

    if len(fault_indices) > 0:
        fault_detected = True
        fault_time = timestamps[fault_indices[0]]  # Time of the first fault
        return fault_detected, fault_time
    return fault_detected, None

def detect_fault_with_thresholds(u0, i0, timestamps, u0_threshold, i0_threshold):
    """
    Detects a fault based on zero-sequence voltage and current thresholds.

    Parameters:
    - u0: Zero-sequence voltage (1D array).
    - i0: Zero-sequence current (1D array).
    - timestamps: Array of timestamps corresponding to the signals.
    - u0_threshold: Threshold for zero-sequence voltage.
    - i0_threshold: Threshold for zero-sequence current.

    Returns:
    - fault_time: Timestamp when the fault is first detected, or None if no fault.
    """
    fault_detected = False
    fault_indices = []

    # Iterate through the length of u0 (assuming u0, i0, and timestamps are of the same length)
    fault_indices = np.where((np.abs(u0) > u0_threshold) & (np.abs(i0) > i0_threshold))[0]
    

    if len(fault_indices) > 0:
        fault_detected = True
        fault_time = timestamps[fault_indices[0]]  # Time of the first fault
        return fault_detected, fault_time
    return fault_detected, None

def detect_fault_Fifth_Harmonic(voltages, currents, time, V0_magnitude, V0_phase, I0_magnitude, I0_phase,threshold_V0,threshold_I0,threshold_Q):
    
    phi_5 = V0_phase - I0_phase  # Phase angle difference
    Q_5 = V0_magnitude * I0_magnitude * np.sin(phi_5)
    
    #threshold_V0 = max(0.1 * np.max(V0_magnitude), min_threshold_V0)
    #threshold_I0 = max(0.1 * np.max(I0_magnitude), min_threshold_I0)
    
    fault_time_idx = np.where(((voltages[:, 0] + voltages[:, 1] + voltages[:, 2]) / 3 >= threshold_V0) &
                              ((currents[:, 0] + currents[:, 1] + currents[:, 2]) / 3 >= threshold_I0))[0]
    
    if len(fault_time_idx) > 0:
        fault_detected = True
        fault_time = time[fault_time_idx[0]]  
    else:
        fault_detected = False
        fault_time = 0
    #fault direction detection based on reactive power and sin_phi
    if Q_5 > threshold_Q:  # Increase the threshold for a Forward fault
        fault_direction = "Forward"
    elif Q_5 < -threshold_Q:  # Increase the threshold for a Reverse fault
        fault_direction = "Reverse"
    else:
        fault_direction = "None"
        fault_detected = False  # Ensure we mark it as no fault
        fault_time = time[0]  # Use the first timestamp for no fault

    return fault_detected,Q_5, fault_direction, fault_time

def classify_wattmetric(u0, i0, timestamps, start_time, threshold=0):
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
    start_index = (np.searchsorted(timestamps, (start_time)))


    # Define the end of the 3-second window
    end_time = start_time + 3.0
    end_index = (np.searchsorted(timestamps, (end_time)))



    # Extract data within the 3-second window
    u0_window = u0[start_index:end_index]
    i0_window = i0[start_index:end_index]

    # Calculate magnitudes and phase difference
    u0_magnitude = np.abs(u0_window)
    i0_magnitude = np.abs(i0_window)
    phase_diff = np.angle(u0_window) - np.angle(i0_window)
    # print("Phase Difference: ", phase_diff)

    phi_threshold = 0.2 * np.mean(phase_diff)
    active_power = u0_magnitude * i0_magnitude * np.cos(phase_diff)
    adjusted_threshold = 2

    fault_type = "Deadzone!!"  # Default state
    for i, power in enumerate(active_power):
        if (phase_diff > phi_threshold).any() and (
                power > adjusted_threshold).any():  # If power exceeds the threshold in positive direction
            fault_type = "Reverse"
            break
        elif (phase_diff > phi_threshold).any() and (
                power < -adjusted_threshold).any():  # If power exceeds the negative threshold
            fault_type = "Forward"
            break

    return fault_type
