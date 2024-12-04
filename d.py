import numpy as np
import matplotlib.pyplot as plt

# File paths (replace with your actual file paths)
cfg_file = r"C:\Users\VISHNU\PycharmProjects\pythonProject5\RESPE Comtrade Data\Fault RESPE 4.cfg"
dat_file = r"C:\Users\VISHNU\PycharmProjects\pythonProject5\RESPE Comtrade Data\Fault RESPE 4.dat"

# Step 1: Parse the `.cfg` file
cfg_metadata = {}
analog_channel_labels = []
digital_channel_labels = []

try:
    with open(cfg_file, "r", encoding="utf-8", errors="replace") as cfg:
        lines = cfg.readlines()

        # Extract metadata
        try:
            cfg_metadata['station_name'], cfg_metadata['rec_dev_id'], cfg_metadata['rev_year'] = lines[0].strip().split(",")
        except ValueError:
            raise ValueError("Error parsing metadata from the first line of the .cfg file.")

        second_line = lines[1].strip().split(",")
        analog_count = int(''.join(filter(str.isdigit, second_line[1])))
        digital_count = int(''.join(filter(str.isdigit, second_line[2])))

        # Extract analog channel labels
        for i in range(2, 2 + analog_count):
            parts = lines[i].strip().split(",")
            analog_channel_labels.append(parts[1])  # Channel name
except FileNotFoundError:
    print(f"Error: The file {cfg_file} was not found.")
    exit(1)

# Step 2: Parse the `.dat` file
timestamps = []
analog_samples = []
digital_samples = []
skipped_lines = 0

try:
    with open(dat_file, "r", encoding="utf-8", errors="replace") as dat:
        for line in dat:
            parts = line.strip().split(",")
            if len(parts) >= 2 + analog_count + digital_count:
                try:
                    timestamps.append(float(parts[1]))
                    analog_samples.append([float(x) for x in parts[2:2 + analog_count]])
                    digital_samples.append([int(x) for x in parts[2 + analog_count:]])
                except ValueError:
                    skipped_lines += 1
except FileNotFoundError:
    print(f"Error: The file {dat_file} was not found.")
    exit(1)

# Logging skipped lines
if skipped_lines > 0:
    print(f"Skipped {skipped_lines} invalid lines in the .dat file.")

# Convert lists to numpy arrays
if timestamps and analog_samples:
    timestamps = np.array(timestamps)
    analog_samples = np.array(analog_samples)
    digital_samples = np.array(digital_samples)
else:
    print("Error: No valid data found in the .dat file.")
    exit(1)

# Step 3: Apply Scaling (if needed)
# Adjust scaling_factor based on actual units (1 if no scaling is needed)
scaling_factor_voltage = 1000  # Example: Convert high voltage to kilovolts
scaling_factor_current = 1  # Example: No scaling needed for currents

scaled_analog_samples = analog_samples.copy()
scaled_analog_samples[:, :3] /= scaling_factor_voltage  # Scale voltage channels
scaled_analog_samples[:, 3:] /= scaling_factor_current  # Scale current channels

# Step 4: Plot Voltage and Current Graphs
plt.figure(figsize=(15, 12))

# Voltage plot
plt.subplot(2, 1, 1)
for i, label in enumerate(analog_channel_labels[:3]):  # First 3 are voltage channels
    plt.plot(timestamps, scaled_analog_samples[:, i], label=label)
plt.title("Voltage Signals")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (kV)")
plt.legend()
plt.grid()

# Current plot
plt.subplot(2, 1, 2)
for i, label in enumerate(analog_channel_labels[3:7]):  # Channels 4 to 7 are currents
    plt.plot(timestamps, scaled_analog_samples[:, i + 3], label=label)  # Offset correctly
plt.title("Current Signals")
plt.xlabel("Time (s)")
plt.ylabel("Current (A)")
plt.legend()
plt.grid()

# Show the plots
plt.tight_layout()
plt.show()
