import numpy as np
import struct

# File paths

cfg_file = r"C:\Users\VISHNU\PycharmProjects\pythonProject5\RESPE Comtrade Data\Fault RESPE 25.cfg"
dat_file = r"C:\Users\VISHNU\PycharmProjects\pythonProject5\RESPE Comtrade Data\Fault RESPE 25.dat"
# Parse the .cfg file
print("Reading Configuration File (.cfg):\n")

cfg_metadata = {}
analog_channel_labels = []
digital_channel_labels = []
with open(cfg_file, "r", encoding="latin-1") as cfg:
    lines = cfg.readlines()
    cfg_metadata['station_name'], cfg_metadata['rec_dev_id'], cfg_metadata['rev_year'] = lines[0].strip().split(",")
    second_line = lines[1].strip().split(",")
    cfg_metadata['total_channels'] = int(second_line[0])
    analog_count = int(''.join(filter(str.isdigit, second_line[1])))
    digital_count = int(''.join(filter(str.isdigit, second_line[2])))
    for i in range(2, 2 + analog_count):
        analog_channel_labels.append(lines[i].strip().split(",")[1])
    for i in range(2 + analog_count, 2 + analog_count + digital_count):
        digital_channel_labels.append(lines[i].strip().split(",")[1])

print("Metadata from .cfg file:")
print(cfg_metadata)
print("Analog Channel Labels:", analog_channel_labels)
print("Digital Channel Labels:", digital_channel_labels)

# Parse the binary .dat file
print("\nReading Data File (.dat):\n")

timestamps = []
analog_samples = []
digital_samples = []

record_format = f"<f{analog_count}f{digital_count}H"
record_size = struct.calcsize(record_format)

with open(dat_file, "rb") as dat:
    while True:
        record = dat.read(record_size)
        if not record:
            break
        try:
            decoded = struct.unpack(record_format, record)
            timestamp = decoded[0]
            analog = decoded[1:1 + analog_count]
            digital = decoded[1 + analog_count:]

            timestamps.append(timestamp)
            analog_samples.append(analog)
            digital_samples.append(digital)
        except struct.error as e:
            print(f"Error unpacking record: {e}")
            break

timestamps = np.array(timestamps)
analog_samples = np.array(analog_samples)
digital_samples = np.array(digital_samples)

# Debugging parsed data
print("\nDebugging Raw Data:")
print("First 5 Analog Samples (Raw):", analog_samples[:5])
print("First 5 Digital Samples (Raw):", digital_samples[:5])
print("Timestamps (First 5):", timestamps[:5])

# Fault Detection Logic
def detect_fault(U0, I0, threshold):
    phase_angle = np.angle(U0)
    cos_phi = np.cos(phase_angle)
    W = U0 * I0 * cos_phi
    if W > threshold:
        return "Reverse Fault Detected"
    elif W < -threshold:
        return "Forward Fault Detected"
    else:
        return "No Fault Detected"

U0 = np.mean(analog_samples[:, :3], axis=1)
I0 = np.mean(analog_samples[:, 3:6], axis=1)

print("\nDebugging Zero-Sequence Computation:")
print("U0 (First 5):", U0[:5])
print("I0 (First 5):", I0[:5])

# Fault Detection
THRESHOLD = 0.1  # Adjusted threshold for debugging
fault_results = [detect_fault(U0[i], I0[i], THRESHOLD) for i in range(len(U0))]

# Debugging Fault Detection Logic
print("\nFault Detection Debugging:")
for i in range(min(5, len(fault_results))):
    W = U0[i] * I0[i] * np.cos(np.angle(U0[i]))
    print(f"Index {i}: U0 = {U0[i]:.3f}, I0 = {I0[i]:.3f}, Power = {W:.3f}, Fault = {fault_results[i]}")

# Display Results
print("\nFault Results (First 5):")
for i in range(min(5, len(fault_results))):
    print(f"Timestamp: {timestamps[i]:.3f}, Fault Direction: {fault_results[i]}")
