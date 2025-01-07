
import os
import numpy as np
import matplotlib.pyplot as plt
from utilities.signal_processing import high_pass_filter, band_pass_filter
from utilities.fault_classification import detect_fault_with_thresholds, classify_fault
from utilities.threshold_compute import compute_thresholds
import pyComtrade

def tune_band_pass_filter(cfg_file, voltage, zero_curr, u0_threshold, i0_threshold, timestamps, fs, expected_results, low_range, high_range, step):
    """
    Systematically tunes the band-pass filter to find optimal cutoff frequencies.

    Args:
        data (array): Input data to be filtered.
        timestamps (array): Time array corresponding to the data.
        fs (float): Sampling frequency (Hz).
        fault_types (list): Detected fault types for each frequency range.
        expected_results (list): Expected fault results for validation.
        low_range (tuple): (min, max) for lower cutoff frequency.
        high_range (tuple): (min, max) for higher cutoff frequency.
        step (float): Step size for tuning.

    Returns:
        tuple: Optimal low and high cutoff frequencies.
    """
    best_match = 0
    best_freqs = (None, None)
    scores = {}

    for low_cutoff in np.arange(low_range[0], low_range[1], step):
        for high_cutoff in np.arange( high_range[0], high_range[1], step):
            if high_cutoff <= low_cutoff:
                continue

            # Apply the band-pass filter
            filtered_voltage = band_pass_filter(voltage, low_cutoff, high_cutoff, fs)
            filtered_curr = band_pass_filter(zero_curr, low_cutoff, high_cutoff, fs)

            # Compute zero-sequence voltage (U0)
            u0 = np.mean(filtered_voltage, axis=0)

            # Detect faults
            detected_faults = detect_fault_with_thresholds(u0, filtered_curr, timestamps, u0_threshold, i0_threshold)
            if detected_faults[0]:
                # Classify the fault
                fault_type = classify_fault(u0, filtered_curr)
                if cfg_file == "rspe4.cfg":
                    match_score = int(fault_type == expected_results[0])
                if cfg_file == "rspe25.cfg":
                    match_score = int(fault_type == expected_results[1])
                if cfg_file == "rspe28.cfg":
                    match_score = int(fault_type == expected_results[2])
                if cfg_file == "rspe29.cfg":
                    match_score = int(fault_type == expected_results[3])


                scores[(low_cutoff, high_cutoff)] = match_score
    # Find the best frequency pair with the highest score
    best_freqs = max(scores, key=scores.get) if scores else (None, None)
    best_score = scores.get(best_freqs, 0)

    print(f"Best Match for {cfg_file}: Low Cutoff = {best_freqs[0]} Hz, High Cutoff = {best_freqs[1]} Hz, Score = {best_score}")
    return scores
# expected_results = ["Forward Fault", "Forward Fault", "Forward Fault", "Reverse Fault"]  # Replace with ground truth
# low_range = (10.0, 100.0)  # Example: Explore lower cutoffs between 10 Hz and 100 Hz
# high_range = (100.0, 1000.0)  # Example: Explore higher cutoffs between 100 Hz and 1 kHz
# tune_band_pass_filter(cfg_file, voltages, zero_seq_current, u0_threshold, i0_threshold, timestamps, sampling_rate, expected_results, low_range, high_range, 10)