import os
import matplotlib.pyplot as plt
import pyComtrade

# Plotting utility
def plot_comtrade_signals(timestamps, signals, signal_labels, title="Signal Plot", ylabel="Amplitude"):
    """
    Plots multiple signals on the same graph with appropriate labels.
    """
    plt.figure(figsize=(12, 6))
    for i, signal in enumerate(signals):
        plt.plot(timestamps, signal, label=signal_labels[i], linewidth=2)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Main COMTRADE processing function
def process_comtrade_data(folder_path, cutoff_freq=20.0):
    """
    Processes COMTRADE files in a folder and identifies faults based on U0 and I0 thresholds.
    """
    # Get list of all .cfg files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.cfg')]
    i = 0
    for cfg_file in files:
        i += 1
        dat_file = cfg_file.replace('.cfg', '.dat')
        if not os.path.exists(os.path.join(folder_path, dat_file)):
            print(f"Dat file not found for {cfg_file}. Skipping.")
            continue

        # Load COMTRADE data
        comtrade = pyComtrade.ComtradeRecord()
        comtrade.read(os.path.join(folder_path, cfg_file), os.path.join(folder_path, dat_file))

        timestamps = comtrade.get_timestamps()

        # Extract raw voltages and zero-sequence current
        raw_voltages = [ch['values'] for ch in comtrade.cfg_data['A'][0:3]]  # Channels 1, 2, 3 for voltages
        raw_zero_seq_current = comtrade.cfg_data['A'][3]['values']  # Channel 4 for zero-sequence current

        # Added part to extract and plot currents (I):

        raw_currents = [ch['values'] for ch in comtrade.cfg_data['A'][4:7]]  # Channels 5, 6, 7 for currents


        min_length = len(timestamps)
        voltages = [v[:min_length] for v in raw_voltages]
        zero_seq_current = raw_zero_seq_current[:min_length] if raw_zero_seq_current is not None else None

        # Adjust current lengths

        currents = [i[:min_length] for i in raw_currents]  # Ensure currents match timestamps

        # Plot raw voltages
        plot_comtrade_signals(
            timestamps, voltages,
            ["Voltage U1", "Voltage U2", "Voltage U3"],
            title="Raw Voltage Signals",
            ylabel="Voltage (V)"
        )

        # Plot raw currents (I)
        plot_comtrade_signals(
            timestamps, currents,
            ["Current I1", "Current I2", "Current I3"],
            title="Raw Current Signals",
            ylabel="Current (A)"
        )



# Define the folder path to your COMTRADE data
folder_path = "/Users/abderrahim/Downloads/RESPE Comtrade Data-2"

# Process the COMTRADE files
process_comtrade_data(folder_path)
