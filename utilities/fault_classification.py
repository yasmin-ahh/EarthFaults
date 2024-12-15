import numpy as np
import matplotlib.pyplot as plt
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