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
    csv_files = sorted(glob.glob(os.path.join(input_dir, "*_cleaned.csv")))
    all_dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            all_dfs.append(df)
        except Exception as e:
            print(f"❌ Failed to read {f}: {e}")

    if not all_dfs:
        print("❌ No valid CSV files found in the input directory.")
        return

    print(f"📂 Loaded {len(all_dfs)} cleaned CSV(s)")

    all_frames = sorted(set().union(*[df["frame_number"].unique() for df in all_dfs]))
    pose_columns = [col for col in all_dfs[0].columns if col not in ["frame_number", "ball_x", "ball_y", "ball_confidence"]]

    trimmed_mean_rows = []

    for frame_num in all_frames:
        row = {"frame_number": frame_num}
        for col in pose_columns:
            values = []
            for df in all_dfs:
                frame_data = df[df["frame_number"] == frame_num]
                if not frame_data.empty and col in frame_data:
                    val = frame_data[col].values[0]
                    if not pd.isna(val):
                        values.append(val)
            if len(values) >= 3:
                row[col] = trim_mean(values, proportiontocut=0.1)
            else:
                row[col] = np.nan

        trimmed_mean_rows.append(row)
        if frame_num % 50 == 0:
            print(f"⏳ Processed frame {frame_num}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(trimmed_mean_rows).to_csv(output_path, index=False)
    print(f"✅ Trimmed mean CSV saved to: {output_path}")
