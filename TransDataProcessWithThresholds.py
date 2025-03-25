import os
import pandas as pd
import numpy as np
from utilities.signal_processing import high_pass_filter, band_pass_filter
from utilities.fault_classification import detect_fault_with_thresholds, classify_fault
from utilities.threshold_compute import compute_thresholds

input_folder_vf = 'newFaultsData/VF/'
input_folder_gf = 'newFaultsData/GF/'
input_folder_hf = 'newFaultsData/HF/'
input_folder_gaf = 'newFaultsData/GAF/'

# Paths
base_folder = "newFaultsData"  # Update this with your actual output folder path
output_csv = "results_tr_with_threshold.csv"

# Fault cases mapping
fault_folders = {
    "VF": os.path.join(base_folder, "VF"),
    "HF": os.path.join(base_folder, "HF"),
    "GF": os.path.join(base_folder, "GF"),
    "GAF": os.path.join(base_folder, "GAF"),
}

# Define sampling rate for filtering
sampling_rate = 10e3  # 10 kHz

# Generate 5 random U0 and I0 threshold values
u0_threshold_values = np.array([0.1*20e3, 0.2*20e3, 0.3*20e3, 0.4*20e3])
i0_threshold_values = np.array([20, 30,40, 50])

# List to store results
detection_results = []

# Process each fault folder
for u0_threshold in u0_threshold_values:
    for i0_threshold in i0_threshold_values:
        for fault_type, folder_path in fault_folders.items():
            if not os.path.exists(folder_path):
                continue

            # Process each CSV file in the folder
            for file_name in os.listdir(folder_path):
                if not file_name.endswith(".csv"):
                    continue

                file_path = os.path.join(folder_path, file_name)
                df = pd.read_csv(file_path)

                fault_case_number = os.path.splitext(file_name)[0]  # Remove extension to get filename

                # Extract necessary columns
                time = df["Time"].values
                voltages = df.iloc[:, 1:4].values  # First 3 columns after "Time"
                currents = df.iloc[:, 4:7].values  # Next 3 columns

                # Apply band-pass filtering
                filtered_voltages = np.array([
                    band_pass_filter(v, 10.0, 100.0, sampling_rate, 4, 'butter', False) for v in voltages.T
                ]).T
                filtered_currents = np.array([
                    band_pass_filter(c, 10.0, 100.0, sampling_rate, 4, 'butter', False) for c in currents.T
                ]).T

                # Compute zero-sequence voltage (U0)
                u0 = np.mean(filtered_voltages, axis=1)
                i0 = np.mean(filtered_currents, axis=1)

                # Detect faults
                fault_detected, fault_time = detect_fault_with_thresholds(u0, i0, time, u0_threshold, i0_threshold)

                if fault_detected:
                    # Classify the fault
                    detected_fault_direction = classify_fault(u0, i0)

                    # Find the closest row in the original data
                    fault_index = np.argmin(np.abs(time - fault_time))
                    original_row = df.iloc[fault_index]

                    U0_at_fault = u0[fault_index]
                    I0_at_fault = i0[fault_index]

                    # Compute Peak Values Over Full Time-Series
                    U0_max = np.max(np.abs(u0))  # Peak zero-sequence voltage across all time
                    I0_max = np.max(np.abs(i0))  # Peak zero-sequence current across all time

                    # Compute Rate of Change Over Full Time-Series
                    dU0_dt = np.max(np.abs(np.gradient(u0, time)))  # Max rate of change of U0 over time
                    dI0_dt = np.max(np.abs(np.gradient(i0, time)))  # Max rate of change of I0 over time

                    # Compute FFT for frequency analysis (on the full U0 signal)
                    fft_U0 = np.fft.fft(u0)
                    freqs = np.fft.fftfreq(len(u0), d=(time[1] - time[0]))  # Frequency axis
                    dominant_freq_U0 = freqs[np.argmax(np.abs(fft_U0))]  # Frequency with highest energy

                    # Energy in high-frequency band (above 100 Hz)
                    high_freq_idx = np.where(freqs > 100)  # Consider above 100 Hz
                    high_freq_energy_U0 = np.sum(np.abs(fft_U0[high_freq_idx]) ** 2)

                    # Compute Spectral Entropy
                    power_spectrum = np.abs(fft_U0) ** 2
                    power_spectrum /= np.sum(power_spectrum)  # Normalize
                    spectral_entropy_U0 = -np.sum(power_spectrum * np.log2(power_spectrum + 1e-10))

                    # Store result
                    detection_results.append([
                        fault_case_number, fault_time, detected_fault_direction, fault_type,
                        # Original direction is the folder name
                        original_row.iloc[1], original_row.iloc[2], original_row.iloc[3],  # Voltages
                        original_row.iloc[4], original_row.iloc[5], original_row.iloc[6],  # Currents
                        U0_at_fault, I0_at_fault,# U0 and I0
                        u0_threshold, i0_threshold ,0,0
                        U0_max, I0_max, dU0_dt, dI0_dt,  # Peak & rate of change features
                        dominant_freq_U0, high_freq_energy_U0, spectral_entropy_U0,  # Frequency features
                        0, 0, 0,  # Wattmetric features (filled with 0)
                        0, 0,  # 5th harmonic features (set to 0)
                        "transient"  # Algorithm used
                    ])
                else:  # No fault detectedDD
                    detected_fault_direction = "None"
                    original_row = df.iloc[0]
                    # Extract voltages & currents from the first row
                    V1, V2, V3 = original_row.iloc[1], original_row.iloc[2], original_row.iloc[3]
                    I1, I2, I3 = original_row.iloc[4], original_row.iloc[5], original_row.iloc[6]

                    # Compute steady-state U0 & I0
                    U0_no_fault = np.mean([V1, V2, V3])
                    I0_no_fault = np.mean([I1, I2, I3])

                    detection_results.append([
                        fault_case_number, -1, detected_fault_direction, fault_type,
                        # Original direction is the folder name
                        V1, V2, V3,  # Use steady-state voltages
                        I1, I2, I3,  # Use steady-state currents
                        U0_no_fault, I0_no_fault,# U0 and I0
                        u0_threshold, i0_threshold,0,0
                        0, 0, 0, 0,  # Transient features (set to 0)
                        0, 0, 0,  # FFT-based features (set to 0)
                        0, 0, 0,  # Wattmetric features (filled with 0)
                        0, 0,  # 5th harmonic features (set to 0)
                        "transient"  # Algorithm used
                    ])

# Convert results to DataFrame and save
df_results = pd.DataFrame(detection_results, columns=[
    "FaultCaseNumber", "timeDetected", "DetectedFaultDirection", "OriginalDirection",
    "RawVoltage1", "RawVoltage2", "RawVoltage3",
    "RawCurrent1", "RawCurrent2", "RawCurrent3",
    "U0", "I0",  # Added zero-sequence voltage & current
    "U0Threshold", "I0Threshold", "Power Threshold", "Q_threshold",
    "U0_max", "I0_max", "dU0_dt", "dI0_dt",  # Peak & rate of change
    "dominant_freq_U0", "high_freq_energy_U0", "spectral_entropy_U0",  # Frequency features
    "ActivePower", "ReactivePower", "PhaseDifference",  # Wattmetric features
    "Q5", "PhaseDifference_5th",  # 5th harmonic features
    "algorithm_used"
])
df_results.to_csv(output_csv, index=False)

print(f"Fault detection results saved to {output_csv}.")


