# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 18:39:56 2025

@author: PSP19
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utilities.signal_processing import extract_fifth_harmonic1
from utilities.fault_classification import detect_fault_Fifth_Harmonic

# Load the Excel file
file_path = 'comdata/SimulationResults_3_seconds_D-optimal.xlsx'
excel_data = pd.ExcelFile(file_path)

# Loop through scenario sheets
for i in range(1, 31):
    sheet_name = f"Scenario_{i}"

    # Load the current scenario sheet
    scenario_data = excel_data.parse(sheet_name)

    # Extract columns into arrays
    timestamps = scenario_data['Time'].to_numpy()
    voltages = scenario_data[['Voltage_A_VF', 'Voltage_B_VF', 'Voltage_C_VF']].to_numpy()
    currents = scenario_data[['Current_A_VF', 'Current_B_VF', 'Current_C_VF']].to_numpy()

    sampling_rate = 10000

    # Extract fifth harmonics
    voltages5 = extract_fifth_harmonic1(voltages, sampling_rate)
    currents5 = extract_fifth_harmonic1(currents, sampling_rate)

    # Compute zero-sequence voltage (U0)
    u0 = np.mean(voltages5, axis=1)
    i0 = np.mean(currents5, axis=1)

    # Detect fault
    result = detect_fault_Fifth_Harmonic(u0, i0, timestamps)
   
    phase_angle = np.angle(u0) - np.angle(i0)
    sin_phi = np.sin(phase_angle)
    power_data = (u0) * (i0) * sin_phi *10e11 # Reactive power
    
    if result[0]:
        # Classify the fault
        fault_direction = result[2]
        print(f"Scenario {i}: Fault detected at {result[1]:.6f} seconds. Fault Direction={fault_direction}.")
    else:
        print(f"Scenario {i}: No fault detected.")

'''
# Plot voltages against timestamps
plt.figure(figsize=(20, 6))
plt.plot(timestamps, voltages5[:, 0], label='Voltage A (VF)', color='red')
plt.plot(timestamps, voltages5[:, 1], label='Voltage B (VF)', color='blue')
plt.plot(timestamps, voltages5[:, 2], label='Voltage C (VF)', color='green')
plt.title('Voltages vs Timestamps')
plt.xlabel('Timestamps (s)')
plt.ylabel('Voltages (V)')
plt.legend()
plt.grid(True)
plt.show()

# Plot voltages against timestamps
plt.figure(figsize=(20, 6))
plt.plot(timestamps, currents5[:, 0], label='Current A (VF)', color='blue')
plt.plot(timestamps, currents5[:, 1], label='Current B (VF)', color='yellow')
plt.plot(timestamps, currents5[:, 2], label='Current C (VF)', color='red')
plt.title('Current vs Timestamps')
plt.xlabel('Timestamps (s)')
plt.ylabel('Current (V)')
plt.legend()
plt.grid(True)
plt.show()

# Plot U0 and I0 signals
plt.figure(figsize=(20, 6))
#plt.plot(timestamps, u0, label="Zero-sequence Voltage (|U0|)")
plt.plot(timestamps, i0, label="Zero-sequence Current (|I0|)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.title('Zero-sequence Voltage and Current')
plt.grid()
plt.show()
'''
