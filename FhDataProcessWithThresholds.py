import os
import pandas as pd
import numpy as np
from utilities.signal_processing import compute_5th_harmonic_zero_sequence
from utilities.fault_classification import detect_fault_Fifth_Harmonic

input_folder_vf = 'newFaultsData/VF/'
input_folder_gf = 'newFaultsData/GF/'
input_folder_hf = 'newFaultsData/HF/'
input_folder_gaf = 'newFaultsData/GAF/'

# Paths
base_folder = "newFaultsData"  # Update this with your actual output folder path
output_csv = "results_fh_with_threshold.csv"

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
Q_threshold_values = np.array([0.5, 1, 1.5,2])
# List to store results
detection_results = []

# Process each fault folder
for u0_threshold in u0_threshold_values:
    for i0_threshold in i0_threshold_values:
        for Q_threshold in Q_threshold_values:
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
                    # Compute the sampling frequency
                    sampling_interval = 10000  # Assuming uniform sampling
                    fs = 1 / sampling_interval  # Sampling frequency
    
                    # Compute 5th harmonic zero-sequence components
                    results = compute_5th_harmonic_zero_sequence(voltages, currents, fs)
                    # Compute fault parameters
                    fault_detected, Q_5, fault_direction, fault_time = detect_fault_Fifth_Harmonic(voltages, currents, time,
                                                                                                   results['V0_magnitude'],
                                                                                                   results['V0_phase'],
                                                                                                   results['I0_magnitude'],
                                                                                                   results['I0_phase'],u0_threshold,i0_threshold,Q_threshold)
    
                    if fault_detected:
                        fault_index = np.argmin(np.abs(time - fault_time))
    
                        # Extract values at the fault time
                        U0_at_fault = results["V0_magnitude"]
                        I0_at_fault = results["I0_magnitude"]
                        phase_diff_5th = results["V0_phase"] - results["I0_phase"]
    
                        # Extract corresponding voltage & current values at the fault time
                        original_row = df.iloc[fault_index]
                        V1, V2, V3 = original_row.iloc[1], original_row.iloc[2], original_row.iloc[3]
                        I1, I2, I3 = original_row.iloc[4], original_row.iloc[5], original_row.iloc[6]
    
                        # Store result
                        detection_results.append([
                            fault_case_number, fault_time, fault_direction, fault_type,
                            V1, V2, V3,  # Voltages
                            I1, I2, I3,  # Currents
                            U0_at_fault, I0_at_fault,  # Zero-sequence voltage and current (5th harmonic magnitude)
                            u0_threshold, i0_threshold,Q_threshold,
                            0, 0, 0, 0,  # Transient features (set to 0)
                            0, 0, 0,  # Transient features (set to 0)
                            0, 0, 0,  # Wattmetric features (set to 0)
                            Q_5, phase_diff_5th,  # 5th harmonic features
                            "fifthHarmonic"
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
    
                        # Store result with steady-state values
                        detection_results.append([
                            fault_case_number, -1, fault_direction, fault_type,
                            V1, V2, V3,  # Use steady-state voltages
                            I1, I2, I3,  # Use steady-state currents
                            U0_no_fault, I0_no_fault,  # Use steady-state zero-sequence values
                            u0_threshold, i0_threshold,Q_threshold,
                            0, 0, 0, 0,  # Transient features (set to 0)
                            0, 0, 0,  # Transient features (set to 0)
                            0, 0, 0,  # Wattmetric features (set to 0)
                            0, 0,  # 5th harmonic features (set to 0)
                            "fifthHarmonic"
                        ])

# Convert results to DataFrame and save
df_results = pd.DataFrame(detection_results, columns=[
    "FaultCaseNumber", "timeDetected", "DetectedFaultDirection", "OriginalDirection",
    "RawVoltage1", "RawVoltage2", "RawVoltage3",
    "RawCurrent1", "RawCurrent2", "RawCurrent3",
    "U0", "I0",  # Added zero-sequence voltage & current
    "U0Threshold", "I0Threshold", "Q_threshold",
    "U0_max", "I0_max", "dU0_dt", "dI0_dt",  # Peak & rate of change
    "dominant_freq_U0", "high_freq_energy_U0", "spectral_entropy_U0",  # Frequency features
    "ActivePower", "ReactivePower", "PhaseDifference",  # Wattmetric features
    "Q5", "PhaseDifference_5th",  # 5th harmonic features
    "algorithm_used"
])
df_results.to_csv(output_csv, index=False)

print(f"Fault detection results saved to {output_csv}.")
