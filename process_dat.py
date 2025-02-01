import os
import pandas as pd
import pyComtrade
import numpy as np

# def process_csv_file()


# split the csv file into sperate fault cases and save them in a new folder [done]
# a total of 504 dat data files in csv format are available [done]
# pass each file as an argument to the function 
# the function will read the csv file and classify the fault case as reverse or forward
# the function will be saving a seperate extra csv file which will contain the fault case timings, the
# fault direction and the algorithm used to classify the fault case
# so the new csv file would have 504 rows and [fault time, fault direction, algorithm used, real fault direction] columns
# 

# Define input and output directories
input_dir = "dataFault/"
output_dir = "newFaultData/"

import os
import pandas as pd

# Define input and output paths
excel_file = "dataFault/simulation126Cases.xlsx"  # Update this with your actual file path
output_dir = "newFaultsData/"  # Update this with your desired output directory

# Create base output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the Excel file
xls = pd.ExcelFile(excel_file)

# Define column groups for each fault case
fault_cases = {
    "VF": ["Time", "Voltage_A_VF", "Voltage_B_VF", "Voltage_C_VF", "Current_A_VF", "Current_B_VF", "Current_C_VF"],
    "HF": ["Time", "Voltage_A_HF", "Voltage_B_HF", "Voltage_C_HF", "Current_A_HF", "Current_B_HF", "Current_C_HF"],
    "GF": ["Time", "Voltage_A_GF", "Voltage_B_GF", "Voltage_C_GF", "Current_A_GF", "Current_B_GF", "Current_C_GF"],
    "GAF": ["Time", "Voltage_A_GAF", "Voltage_B_GAF", "Voltage_C_GAF", "Current_A_GAF", "Current_B_GAF", "Current_C_GAF"],
}

# Loop through each sheet in the Excel file
for sheet_name in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet_name)

    # Process each fault case separately
    for case, columns in fault_cases.items():
        if all(col in df.columns for col in columns):  # Ensure columns exist
            case_df = df[columns]

            # Create subdirectory for the fault case
            case_folder = os.path.join(output_dir, case)
            os.makedirs(case_folder, exist_ok=True)

            # Generate output filename
            fault_filename = f"{sheet_name}_{case}.csv"
            output_path = os.path.join(case_folder, fault_filename)

            # Save to CSV
            case_df.to_csv(output_path, index=False)

print("Fault cases separated and saved successfully.")

