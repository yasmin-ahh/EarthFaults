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
        return "Forward Fault"
    else:
        return "Reverse Fault"

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

def detect_fault_Fifth_Harmonic(u0, i0, timestamps,power_data=None):
    """
    Detects a fault based on zero-sequence voltage and current thresholds.

    Parameters:
    - u0: Zero-sequence voltage (1D array).
    - i0: Zero-sequence current (1D array).
    - timestamps: Array of timestamps corresponding to the signals.

    Returns:
    - fault_detected : True or Faule if the fault is detected    
    - fault_time: Timestamp when the fault is first detected, or None if no fault.
    - fault_direction: Forward or Reverse fault based on sin_phi
    """
    fault_detected = False
    fault_indices = []
    
    if power_data is None:
       phase_angle = np.angle(u0) - np.angle(i0)
       sin_phi = np.sin(phase_angle)
       power_data = (u0) * (i0) * sin_phi *10e11  # Reactive power
    '''
    # Plot Reactive power vs Time
    plt.figure(figsize=(20, 6))
    plt.plot(timestamps, power_data, label=" Reactive Power (W)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title("Reactive Power")
    plt.grid()
    plt.show()
    '''
    # Iterate through the length of power data
    fault_indices = np.where(power_data>1)[0]
    

    if len(fault_indices) > 0:
        fault_detected = True
        fault_time = timestamps[fault_indices[0]]  # Time of the first fault
        if sin_phi[fault_indices[0]] < 0:
            fault_direction = "Forward Fault"
        else:
            fault_direction = "Reverse Fault"    
        
        return fault_detected, fault_time, fault_direction
    return fault_detected, None