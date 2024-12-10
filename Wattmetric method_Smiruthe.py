import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt


# Step 1: Define High-Pass Filter
def high_pass_filter(signal, cutoff, fs, order=4):
    """
    Apply a high-pass filter to remove low-frequency noise.
    Args:
        signal (array): Input signal.
        cutoff (float): Cutoff frequency.
        fs (float): Sampling frequency.
        order (int): Filter order.
    Returns:
        array: Filtered signal.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, signal)

# Step 2: Compute Zero-Sequence Components
def compute_zero_sequence(voltages, currents):
    """
    Compute zero-sequence voltage and current components.
    Args:
        voltages (array): Voltage samples [Ua, Ub, Uc].
        currents (array): Current samples [Ia, Ib, Ic].
    Returns:
        tuple: Zero-sequence voltage (U0) and current (I0).
    """
    U0 = np.mean(voltages, axis=1)  # Zero-sequence voltage
    I0 = np.mean(currents, axis=1)  # Zero-sequence current
    return U0, I0

# Step 3: Fault Detection Logic
def detect_fault_with_time(U0, I0, threshold, timestamps):
    """
    Detect faults based on zero-sequence components and active power, and identify fault occurrence time.
    Args:
        U0 (array): Zero-sequence voltage.
        I0 (array): Zero-sequence current.
        threshold (float): Fault detection threshold.
        timestamps (array): Array of timestamps corresponding to U0 and I0.
    Returns:
        tuple: Fault detection results and fault occurrence time.
    """
    phase_angle = np.angle(U0) - np.angle(I0)  # Phase angle of zero-sequence components
    cos_phi = np.cos(phase_angle)  # Power factor
    W = U0 * I0 * cos_phi  # Active power
    fault_time = None

    # Initialize results
    results = "No Fault Detected"

    # Check for the first instance where active power crosses the threshold
    for i, power in enumerate(W):
        if power > threshold:
            results = "Reverse Fault Detected"
            fault_time = timestamps[i]
            break
        elif power < -threshold:
            results = "Forward Fault Detected"
            fault_time = timestamps[i]
            break

    print("\nActive Power : ", np.mean(W))
    print(f"Fault Time: {fault_time if fault_time else 'No Fault Detected'}")
    print("Fault type : ", results)
    return results, fault_time


# Step 4: Calculating Threshold from Fault-free data
def compute_threshold(U0, I0, fault_free_start, fault_free_end, timestamps, n=3):
    """
    Compute threshold for fault detection based on fault-free data.

    Args:
        U0 (array): Zero-sequence voltage array.
        I0 (array): Zero-sequence current array.
        fault_free_start (float): Start time of fault-free region (seconds).
        fault_free_end (float): End time of fault-free region (seconds).
        timestamps (array): Array of timestamps corresponding to U0 and I0.
        n (int): Sensitivity multiplier for threshold calculation.

    Returns:
        float: Calculated threshold.
    """


    # Identify fault-free data indices based on the given time range
    fault_free_indices = (timestamps >= fault_free_start) & (timestamps <= fault_free_end)

    # Extract fault-free data
    fault_free_U0 = U0[fault_free_indices]
    fault_free_I0 = I0[fault_free_indices]

    # Compute active power (W) in fault-free region
    phase_angle = np.angle(fault_free_U0) - np.angle(fault_free_I0)
    cos_phi = np.cos(phase_angle)
    fault_free_W = fault_free_U0 * fault_free_I0 * cos_phi

    # Compute mean and standard deviation of active power
    mean_W = np.mean(fault_free_W)
    std_W = np.std(fault_free_W)

    # Calculate threshold using mean + n * std deviation
    threshold = mean_W + n * std_W
    print(f"Computed Threshold: {threshold}")

    return threshold

# Step 5: COMTRADE File Parsing
cfg_file = r"C:\Users\smiru\OneDrive\Desktop\TUD\TUD - 3\PROJECT\RESPE Comtrade Data\RESPE Comtrade Data\Fault RESPE 4.cfg"
dat_file = r"C:\Users\smiru\OneDrive\Desktop\TUD\TUD - 3\PROJECT\RESPE Comtrade Data\RESPE Comtrade Data\Fault RESPE 4.dat"

# Parse the .cfg file
cfg_metadata = {}
analog_channel_labels = []
digital_channel_labels = []
with open(cfg_file, "r") as cfg:
    lines = cfg.readlines()

    # Extract station name and metadata
    cfg_metadata['station_name'], cfg_metadata['rec_dev_id'], cfg_metadata['rev_year'] = lines[0].strip().split(",")

    # Parse the second line
    second_line = lines[1].strip().split(",")
    cfg_metadata['total_channels'] = int(second_line[0])
    analog_count = int(''.join(filter(str.isdigit, second_line[1])))  # Remove non-numeric characters
    digital_count = int(''.join(filter(str.isdigit, second_line[2])))  # Remove non-numeric characters

    # Extract analog channel labels
    for i in range(2, 2 + analog_count):
        parts = lines[i].strip().split(",")
        analog_channel_labels.append(parts[1])  # Channel name

    # Extract digital channel labels
    for i in range(2 + analog_count, 2 + analog_count + digital_count):
        parts = lines[i].strip().split(",")
        digital_channel_labels.append(parts[1])  # Channel name

    # Extract sampling rates
    sampling_rate_line = lines[-2].strip().split(",")
    cfg_metadata['sampling_rates'] = [float(rate) if rate.replace('.', '', 1).isdigit() else rate for rate in sampling_rate_line]

print("Metadata from .cfg file:")
print(cfg_metadata)
print("Analog Channel Labels:", analog_channel_labels)
print("Digital Channel Labels:", digital_channel_labels)

# Parse the .dat file
timestamps = []
analog_samples = []
digital_samples = []
with open(dat_file, "r") as dat:
    for line in dat:
        parts = line.strip().split(",")

        if len(parts) >= 2 + analog_count + digital_count:
            try:
                timestamps.append(float(parts[1]))  # Timestamp
                analog_samples.append([float(x) for x in parts[2:2 + analog_count]])  # Analog data
                digital_samples.append([int(x) for x in parts[2 + analog_count:]])  # Digital data
            except ValueError:
                print(f"Skipping malformed line: {line.strip()}")
        else:
            print(f"Skipping invalid line (not enough columns): {line.strip()}")

# Convert to numpy arrays
timestamps = np.array(timestamps)
analog_samples = np.array(analog_samples)
digital_samples = np.array(digital_samples)

print("First 5 Timestamps:", timestamps[:5])
print("First 5 Analog Samples:")
print(analog_samples[:5, :])
print("First 5 Digital Samples:")
print(digital_samples[:5, :])

# Step 5: Process Signals for Wattmetric Method
sampling_frequency = 10000  # Example sampling frequency (Hz)
cutoff_frequency = 0.1  # High-pass filter cutoff frequency (Hz)

# Split voltage and current signals
voltages = analog_samples[:, :3]  # Columns for Ua, Ub, Uc
currents = analog_samples[:, 3:6]  # Columns for Ia, Ib, Ic

# Apply high-pass filter to remove low-frequency noise
filtered_voltages = np.array([high_pass_filter(v, cutoff_frequency, sampling_frequency) for v in voltages.T]).T
filtered_currents = np.array([high_pass_filter(c, cutoff_frequency, sampling_frequency) for c in currents.T]).T

# Compute zero-sequence components
U0, I0 = compute_zero_sequence(filtered_voltages, filtered_currents)
print("U0:", U0, " I0: ", I0)

# Compute threshold
# Define fault-free time range (e.g., 0 to 2 seconds)
fault_free_start = 0.0  # Start time in seconds
fault_free_end = 0.3  # End time in seconds
THRESHOLD = compute_threshold(U0, I0, fault_free_start, fault_free_end, timestamps, n=3)

# Detect faults
fault_results = detect_fault_with_time(U0, I0, THRESHOLD,timestamps)

# Plotting the original voltages and currents
plt.figure(figsize=(15, 10))

# Subplot for voltages
plt.subplot(4, 1, 1)
plt.plot(timestamps, voltages[:, 0], label='Ua', color='blue')
plt.plot(timestamps, voltages[:, 1], label='Ub', color='orange')
plt.plot(timestamps, voltages[:, 2], label='Uc', color='green')
plt.title("Three-Phase Voltages")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.legend()
plt.grid()

# Subplot for currents
plt.subplot(4, 1, 2)
plt.plot(timestamps, currents[:, 0], label='Ia', color='red')
plt.plot(timestamps, currents[:, 1], label='Ib', color='purple')
plt.plot(timestamps, currents[:, 2], label='Ic', color='brown')
plt.title("Three-Phase Currents")
plt.xlabel("Time (s)")
plt.ylabel("Current (A)")
plt.legend()
plt.grid()

# Subplot for zero-sequence voltage
plt.subplot(4, 1, 3)
plt.plot(timestamps, U0, label='Zero-sequence Voltage (U0)', color='blue')
plt.title("Zero-Sequence Voltage (U0)")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.legend()
plt.grid()

# Subplot for zero-sequence current
plt.subplot(4, 1, 4)
plt.plot(timestamps, I0, label='Zero-sequence Current (I0)', color='red')
plt.title("Zero-Sequence Current (I0)")
plt.xlabel("Time (s)")
plt.ylabel("Current (A)")
plt.legend()
plt.grid()

# Adjust spacing
plt.tight_layout()

# Show the plot
plt.show()