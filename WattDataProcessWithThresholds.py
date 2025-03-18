import os
import pandas as pd
import numpy as np
from utilities.fault_classification import detect_fault_with_thresholds, classify_wattmetric

input_folder_vf = 'newFaultsData/VF/'
input_folder_gf = 'newFaultsData/GF/'
input_folder_hf = 'newFaultsData/HF/'
input_folder_gaf = 'newFaultsData/GAF/'

# Paths
base_folder = "newFaultsData"  # Update this with your actual output folder path
output_csv = "results_wt_with_threshold.csv"

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
power_threshold_values = np.round(np.random.choice(np.linspace(2, 5, 4), 4, replace=False))
# List to store results
detection_results = []

# Iterate over different threshold values
for u0_threshold in u0_threshold_values:
    for i0_threshold in i0_threshold_values:
        for power_threshold in power_threshold_values:
            # Process each fault folder
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

                    # Compute zero-sequence voltage (U0)
                    u0 = np.mean(voltages, axis=1)
                    i0 = np.mean(currents, axis=1)

                    # Detect faults
                    fault_detected, fault_time = detect_fault_with_thresholds(u0, i0, time, u0_threshold, i0_threshold)

                    if fault_detected:
                        # Classify the fault
                        detected_fault_direction = classify_wattmetric(u0, i0, time, fault_time, power_threshold)

                        # Find the closest row in the original data
                        fault_index = np.argmin(np.abs(time - fault_time))
                        original_row = df.iloc[fault_index]

                        # Extract values at the fault time
                        U0_at_fault = u0[fault_index]
                        I0_at_fault = i0[fault_index]

                        # Extract corresponding voltage & current values at the fault time
                        V1, V2, V3 = original_row.iloc[1], original_row.iloc[2], original_row.iloc[3]
                        I1, I2, I3 = original_row.iloc[4], original_row.iloc[5], original_row.iloc[6]

                        # Compute Wattmetric Features
                        phase_diff = np.angle(V1 + V2 + V3) - np.angle(I1 + I2 + I3)
                        active_power = np.abs(U0_at_fault) * np.abs(I0_at_fault) * np.cos(phase_diff)
                        reactive_power = np.abs(U0_at_fault) * np.abs(I0_at_fault) * np.sin(phase_diff)

                        # Store result
                        detection_results.append([
                            fault_case_number, fault_time, detected_fault_direction, fault_type,
                            # Original direction is the folder name
                            V1, V2, V3,  # Voltages
                            I1, I2, I3,  # Currents
                            U0_at_fault, I0_at_fault,  # Zero-sequence voltage and current
                            u0_threshold, i0_threshold, power_threshold,
                            0, 0, 0, 0,  # Transient features (set to 0)
                            0, 0, 0,  # FFT-based features (set to 0)
                            active_power, reactive_power, phase_diff,  # Wattmetric features
                            0, 0,  # 5th harmonic features (set to 0)
                            "wattmetric"  # Algorithm used
                        ])
                    else:  # No fault detected
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
                            U0_no_fault, I0_no_fault,  # U0 and I0
                            u0_threshold, i0_threshold, power_threshold,
                            0, 0, 0, 0,  # Transient features (set to 0)
                            0, 0, 0,  # FFT-based features (set to 0)
                            0, 0, 0,  # Wattmetric features (filled with 0)
                            0, 0,  # 5th harmonic features (set to 0)
                            "wattmetric"  # Algorithm used
                        ])

# Convert results to DataFrame and save
df_results = pd.DataFrame(detection_results, columns=[
    "FaultCaseNumber", "timeDetected", "DetectedFaultDirection", "OriginalDirection",
    "RawVoltage1", "RawVoltage2", "RawVoltage3",
    "RawCurrent1", "RawCurrent2", "RawCurrent3",
    "U0", "I0",  # Added zero-sequence voltage & current
    "U0Threshold", "I0Threshold", "Power Threshold",
    "U0_max", "I0_max", "dU0_dt", "dI0_dt",  # Peak & rate of change
    "dominant_freq_U0", "high_freq_energy_U0", "spectral_entropy_U0",  # Frequency features
    "ActivePower", "ReactivePower", "PhaseDifference",  # Wattmetric features
    "Q5", "PhaseDifference_5th",  # 5th harmonic features
    "algorithm_used"
])
df_results.to_csv(output_csv, index=False)

print(f"Fault detection results saved to {output_csv}.")
