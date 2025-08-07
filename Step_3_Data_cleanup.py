import os
import glob
import numpy as np
import pandas as pd

def adaptive_zscore_filter_weighted(values, window_size=5, base_z=6.0, std_scale=0.0):
    """
    Adaptive Z-score filtering with weighted averaging for outlier removal.

    Preconditions:
        - `values`: List of numeric values (e.g., x or y coordinates) to clean.
        - `window_size`: The window size for local filtering.
        - `base_z`: The base Z-score threshold.
        - `std_scale`: The scaling factor for the standard deviation.
        
    Postconditions:
        - Returns the cleaned list of values with outliers replaced by a weighted average.
    """
    cleaned = []
    values = np.array(values, dtype=np.float32)

    for i in range(len(values)):
        start = max(0, i - window_size + 1)
        window = values[start:i+1]

        if len(window) < 3:
            cleaned.append(values[i])
            continue

        mean = np.nanmean(window)
        std = np.nanstd(window)

        if std == 0 or np.isnan(values[i]):
            cleaned.append(values[i])
            continue

        z_thresh = base_z + (std * std_scale)

        if abs(values[i] - mean) / std < z_thresh:
            cleaned.append(values[i])
        else:
            weights = np.linspace(1, 2, len(window))
            weighted_avg = np.average(window, weights=weights)
            cleaned.append(weighted_avg)

    return cleaned

def process_csv_cleanup(input_dir, output_dir):
    """
    Processes each CSV file, applies data cleanup, and saves the cleaned data.
    
    Preconditions:
        - `input_dir`: Directory containing CSV files to clean.
        - `output_dir`: Directory to save the cleaned CSV files.
        
    Postconditions:
        - CSV files are processed, cleaned, and saved in the output directory.
    """
    joints = [
        "shoulder_left", "shoulder_right", "elbow_left", "elbow_right",
        "wrist_left", "wrist_right", "hip_left", "hip_right",
        "knee_left", "knee_right", "ankle_left", "ankle_right"
    ]
    coords = ["x", "y", "rel_x", "rel_y"]
    marker_columns = [f"{joint}_{coord}" for joint in joints for coord in coords] + ["ref_x", "ref_y"]

    for file_path in glob.glob(os.path.join(input_dir, "*.csv")):
        print(f"\nProcessing {file_path}...")
        df = pd.read_csv(file_path)

        for col in marker_columns:
            if col in df.columns:
                print(f"  → Cleaning column: {col}")
                cleaned_values = adaptive_zscore_filter_weighted(df[col].values)
                df[f"{col}_cleaned"] = cleaned_values
            else:
                print(f"  [!] Skipped missing column: {col}")

        # Save cleaned CSV
        base_name = os.path.basename(file_path)
        save_path = os.path.join(output_dir, base_name.replace(".csv", "_cleaned.csv"))
        df.to_csv(save_path, index=False)
        print(f"✔ Saved cleaned file to: {save_path}")
