import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import trim_mean

def trim_mean_poses(input_dir, output_path):
    """
    Computes the trimmed mean pose data across all CSV files for each frame.
    
    Preconditions:
        - `input_dir`: Directory containing the cleaned CSV files.
        - `output_path`: Path to save the trimmed mean pose CSV file.
        
    Postconditions:
        - The trimmed mean pose data is computed and saved to the specified CSV file.
    """
    # Load all the cleaned CSVs
    csv_files = sorted(glob.glob(os.path.join(input_dir, "*_cleaned.csv")))
    all_dfs = [pd.read_csv(f) for f in csv_files]
    print(f"📂 Loaded {len(all_dfs)} CSVs")

    # Get all unique frame numbers
    all_frames = sorted(set().union(*[df["frame_number"].unique() for df in all_dfs]))

    # Columns for the pose data
    pose_columns = [col for col in all_dfs[0].columns if col not in ["frame_number", "ball_x", "ball_y", "ball_confidence"]]

    # Build trimmed mean frame-by-frame
    trimmed_mean_rows = []

    for frame_num in all_frames:
        row = {"frame_number": frame_num}
        for col in pose_columns:
            values = [df[df["frame_number"] == frame_num][col].values[0]
                      for df in all_dfs
                      if frame_num in df["frame_number"].values and not pd.isna(df[df["frame_number"] == frame_num][col].values[0])]
            if len(values) >= 3:
                row[col] = trim_mean(values, proportiontocut=0.1)
            else:
                row[col] = np.nan  # Not enough data to trim

        trimmed_mean_rows.append(row)

    # Save to CSV
    trimmed_df = pd.DataFrame(trimmed_mean_rows)
    trimmed_df.to_csv(output_path, index=False)
    print(f"✅ Trimmed mean CSV saved to: {output_path}")
