
import numpy as np

def calculate_nominal_voltage(cfg_channel):
    """
    Calculates the nominal voltage for a given channel based on its primary/secondary ratio and scaling factor.

    Parameters:
    - cfg_channel: Dictionary containing channel configuration.

    Returns:
    - Nominal voltage (float).
    """
    primary = cfg_channel['primary']  # PT primary voltage (from the cfg)
    secondary = cfg_channel['secondary']  # PT secondary voltage (from the cfg)
    scaling_factor = cfg_channel['a']  # Scaling factor for the channel

    # Calculate nominal voltage
    nominal_voltage = (primary / secondary) * (1 / scaling_factor)
    return nominal_voltage


def compute_thresholds(cfg_data, zero_seq_current):
    """
    Computes automated thresholds for zero-sequence voltage (U0) and current (I0).

    Parameters:
    - cfg_data: Configuration data extracted from the COMTRADE file.
    - zero_seq_current: Zero-sequence current (I0) values.

    Returns:
    - u0_threshold: Threshold for zero-sequence voltage.
    - i0_threshold: Threshold for zero-sequence current.
    """
    # Calculate nominal voltages for each phase
    nominal_voltages = [
        calculate_nominal_voltage(cfg_data['A'][i]) for i in range(3)  # For U L1-N, U L2-N, U L3-N
    ]
    # Zero-sequence voltage threshold: 20% of the mean nominal voltage
    nominal_voltage = 20e3
    u0_threshold = 0.3 * nominal_voltage
    # Zero-sequence current threshold: Dynamic (mean + 3 * std), minimum of 10A
    abs_i0 = np.abs(zero_seq_current)
    i0_mean = np.mean(zero_seq_current)
    i0_std = np.std(zero_seq_current)
    # i0_threshold =  i0_mean + 3 * i0_std  # Minimum threshold is set to 10A

    i0_threshold = 50

    print(f"Computed Thresholds - U0: {u0_threshold:.2f} V, I0: {i0_threshold} A")  # Debugging
    return u0_threshold, i0_threshold
    
def compute_thresholds_Fifth_Harmonic(u0, i0):
    """
    Computes automated thresholds for zero-sequence voltage (U0) and current (I0).

    Parameters:
    - cfg_data: Configuration data extracted from the COMTRADE file.
    - zero_seq_current: Zero-sequence current (I0) values.

    Returns:
    - u0_threshold: Threshold for zero-sequence voltage.
    - i0_threshold: Threshold for zero-sequence current.
    """
  
    u0_mean = np.mean(u0)
    u0_std = np.std(u0)
    # Zero-sequence voltage threshold: 20% of the mean nominal voltage
    u0_threshold = u0_mean + 3*u0_std
    # Zero-sequence current threshold: Dynamic (mean + 3 * std), minimum of 10A

    i0_mean = np.mean(i0)
    i0_std = np.std(i0)
    i0_threshold =  i0_mean + 3 * i0_std  # Minimum threshold is set to 10A


    print(f"Computed Thresholds - U0: {u0_threshold:.2f} V, I0: {i0_threshold} A")  # Debugging
    return u0_threshold, i0_threshold

def compute_thresholds_wattmetric(cfg_data):
    """
    Computes automated thresholds for zero-sequence voltage (U0) and current (I0).

    Parameters:
    - cfg_data: Configuration data extracted from the COMTRADE file.
    - zero_seq_current: Zero-sequence current (I0) values.

    Returns:
    - u0_threshold: Threshold for zero-sequence voltage.
    - i0_threshold: Threshold for zero-sequence current.
    """
    # Calculate nominal voltages for each phase
    nominal_voltages = [
        calculate_nominal_voltage(cfg_data['A'][i]) for i in range(3)  # For U L1-N, U L2-N, U L3-N
    ]
    mean_nominal_voltage = np.mean(nominal_voltages)
    # Zero-sequence voltage threshold: 20% of the mean nominal voltage
    u0_threshold = 0.3 * mean_nominal_voltage


    print(f"Computed Thresholds - U0: {u0_threshold:.2f} V")  # Debugging
    return u0_threshold
