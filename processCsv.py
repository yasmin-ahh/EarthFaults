import os
import pandas as pd
import pyComtrade
import numpy as np


def process_comtrade_file(cfg_file, dat_file, output_cfg_csv, output_dat_csv):
    """
    Processes a single COMTRADE file and saves .cfg and .dat data into separate CSV files.

    Parameters:
    - cfg_file: Path to the .cfg file.
    - dat_file: Path to the .dat file.
    - output_cfg_csv: Path to the output .cfg CSV file.
    - output_dat_csv: Path to the output .dat CSV file.
    """
    # Read the COMTRADE file (both .cfg and .dat)
    comtradeObj = pyComtrade.ComtradeRecord()
    comtradeObj.read(cfg_file, dat_file)

    # Process .cfg content
    channels = comtradeObj.cfg_data['A']  # Analog channels
    cfg_rows = []
    for idx, channel in enumerate(channels):
        cfg_row = {
            "Channel Number": idx + 1,
            "Channel Name": channel['ch_id'],
            "Type": "Voltage" if "V" in channel['uu'] else "Current",
            "Units": channel['uu'],
            "Scaling Factor (a)": channel['a'],
            "Offset (b)": channel['b'],
            "Primary Voltage/Current": channel['primary'],
            "Secondary Voltage/Current": channel['secondary'],
            "Phase": channel['ph']
        }
        cfg_rows.append(cfg_row)

    # Save .cfg content to CSV
    df_cfg = pd.DataFrame(cfg_rows)
    df_cfg.to_csv(output_cfg_csv, index=False)
    print(f"Saved .cfg content to {output_cfg_csv}")

    # Process .dat content
    timestamps = comtradeObj.get_timestamps()
    dat_data = {"Timestamp": timestamps}

    # Scale and add each channel's data
    for idx, channel in enumerate(channels):
        raw_values = channel['values']
        scaled_values = np.array(raw_values) * channel['a'] + channel['b']
        dat_data[channel['ch_id']] = scaled_values

    # Save .dat content to CSV
    df_dat = pd.DataFrame(dat_data)
    df_dat.to_csv(output_dat_csv, index=False)
    print(f"Saved .dat content to {output_dat_csv}")


def process_comtrade_folder(folder_path, output_cfg_folder, output_dat_folder):
    """
    Processes all COMTRADE files in a folder and saves their contents into separate CSV files.

    Parameters:
    - folder_path: Path to the folder containing .cfg and .dat files.
    - output_cfg_folder: Folder to save .cfg content CSV files.
    - output_dat_folder: Folder to save .dat content CSV files.
    """
    if not os.path.exists(output_cfg_folder):
        os.makedirs(output_cfg_folder)

    if not os.path.exists(output_dat_folder):
        os.makedirs(output_dat_folder)

    files = [f for f in os.listdir(folder_path) if f.endswith('.cfg')]
    for cfg_file in files:
        dat_file = cfg_file.replace('.cfg', '.dat')
        cfg_path = os.path.join(folder_path, cfg_file)
        dat_path = os.path.join(folder_path, dat_file)

        # Output file paths
        cfg_csv = os.path.join(output_cfg_folder, cfg_file.replace('.cfg', '_cfg.csv'))
        dat_csv = os.path.join(output_dat_folder, cfg_file.replace('.cfg', '_dat.csv'))

        # Process both .cfg and .dat
        if os.path.exists(dat_path):
            process_comtrade_file(cfg_path, dat_path, cfg_csv, dat_csv)
        else:
            print(f"Dat file not found for {cfg_file}. Skipping processing.")


# Example usage
folder_path = "comdata"  # Path to COMTRADE files
output_cfg_folder = "csv_cfg"  # Folder for .cfg content CSVs
output_dat_folder = "csv_dat"  # Folder for .dat content CSVs
process_comtrade_folder(folder_path, output_cfg_folder, output_dat_folder)
