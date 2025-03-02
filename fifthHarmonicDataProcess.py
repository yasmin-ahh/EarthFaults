
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
output_csv = "fault_detection_results_fh.csv"

# Fault cases mapping
fault_folders = {
    "VF": os.path.join(base_folder, "VF"),
    "HF": os.path.join(base_folder, "HF"),
    "GF": os.path.join(base_folder, "GF"),
    "GAF": os.path.join(base_folder, "GAF"),
}

# Define sampling rate for filtering
sampling_rate = 10e3  # 10 kHz

# List to store results
detection_results = []

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
        # Compute the sampling frequency
        sampling_interval = 10000  # Assuming uniform sampling
        fs = 1 / sampling_interval  # Sampling frequency
        
        
        # Compute 5th harmonic zero-sequence components
        results = compute_5th_harmonic_zero_sequence(voltages, currents, fs)
        # Compute fault parameters
        fault_detected,Q_5, fault_direction, fault_time = detect_fault_Fifth_Harmonic(voltages, currents, time, 
                                                                    results['V0_magnitude'], results['V0_phase'], 
                                                                    results['I0_magnitude'], results['I0_phase'])

        if fault_detected:
            # Find the closest row in the original data
            fault_index = np.argmin(np.abs(time - fault_time))
            original_row = df.iloc[fault_index]

            # Store result
            detection_results.append([
                fault_case_number, fault_time, fault_direction, fault_type,  # Original direction is the folder name
                original_row.iloc[1], original_row.iloc[2], original_row.iloc[3],  # Voltages
                original_row.iloc[4], original_row.iloc[5], original_row.iloc[6],  # Currents
                "fifthHarmonic"  # Algorithm used
            ])
        else: # No fault detected
            fault_direction = "None"
            detection_results.append([
                fault_case_number, -1, fault_direction, fault_type,  # Original direction is the folder name
                None, None, None,  # Voltages
                None, None, None,  # Currents
                "fifthHarmonic"  # Algorithm used
            ])

# Convert results to DataFrame and save
df_results = pd.DataFrame(detection_results, columns=[
    "FaultCaseNumber", "timeDetected", "DetectedFaultDirection", "OriginalDirection",
    "Voltage1", "Voltage2", "Voltage3",
    "Current1", "Current2", "Current3", "algo_used"
])
df_results.to_csv(output_csv, index=False)

print(f"Fault detection results saved to {output_csv}.")
