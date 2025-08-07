import cv2
import pandas as pd
import numpy as np
import os

# Function to load the CSV data
def load_pose_data(csv_path):
    """
    Load pose data from the given CSV file.
    
    Preconditions:
        - `csv_path`: The path to the CSV file containing pose data.
        
    Postconditions:
        - Returns a DataFrame containing the pose data.
    """
    return pd.read_csv(csv_path)

# Function to normalize the joint coordinates
def normalize_point(x, y, original_w, original_h, target_w, target_h, padding=100):
    """
    Normalize joint coordinates to fit into the target frame size.
    
    Preconditions:
        - `x`, `y`: Joint coordinates in the original frame.
        - `original_w`, `original_h`: The original width and height of the pose data.
        - `target_w`, `target_h`: The target width and height for the frame.
        - `padding`: Padding to be left around the edges of the frame.
        
    Postconditions:
        - Returns the normalized coordinates.
    """
    scale_x = (target_w - 2 * padding) / original_w
    scale_y = (target_h - 2 * padding) / original_h
    x = int(x * scale_x + padding)
    y = int(y * scale_y + padding)
    return x, y

# Function to estimate the original width and height of the pose data
def estimate_original_dimensions(df):
    """
    Estimate the original width and height of the pose data based on the joint coordinates.
    
    Preconditions:
        - `df`: DataFrame containing the pose data with joint coordinates.
        
    Postconditions:
        - Returns the original width (`orig_w`) and height (`orig_h`).
    """
    all_x = df[[c for c in df.columns if c.endswith("_x_cleaned")]].values.flatten()
    all_y = df[[c for c in df.columns if c.endswith("_y_cleaned")]].values.flatten()
    orig_w = np.nanmax(all_x) - np.nanmin(all_x)
    orig_h = np.nanmax(all_y) - np.nanmin(all_y)
    return orig_w, orig_h

# Function to create the video frames with joints and skeleton
def create_frame(row, df, joints, connections, orig_w, orig_h, frame_width, frame_height, bg_color):
    """
    Create a single frame for the video by drawing the joints and skeleton.
    
    Preconditions:
        - `row`: The row of the DataFrame representing a single frame.
        - `df`: The DataFrame containing pose data.
        - `joints`: A list of joint names.
        - `connections`: A list of tuples representing joint connections for the skeleton.
        - `orig_w`, `orig_h`: Original width and height of the pose data.
        - `frame_width`, `frame_height`: Target width and height for the frame.
        - `bg_color`: Background color for the frame.
        
    Postconditions:
        - Returns the generated frame as an image (NumPy array).
    """
    frame = np.full((frame_height, frame_width, 3), bg_color, dtype=np.uint8)
    coords = {}

    # Load joint positions and draw circles
    for joint in joints:
        x_col = f"{joint}_x_cleaned"
        y_col = f"{joint}_y_cleaned"
        if x_col in df.columns and y_col in df.columns:
            try:
                x = float(row[x_col])
                y = float(row[y_col])
                if not np.isnan(x) and not np.isnan(y):
                    norm_x, norm_y = normalize_point(x, y, orig_w, orig_h, frame_width, frame_height)
                    coords[joint] = (norm_x, norm_y)
                    cv2.circle(frame, (norm_x, norm_y), 8, (0, 255, 0), -1)
                    cv2.putText(frame, joint, (norm_x + 5, norm_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            except:
                continue

    # Draw skeleton lines
    for pt1, pt2 in connections:
        if pt1 in coords and pt2 in coords:
            cv2.line(frame, coords[pt1], coords[pt2], (0, 255, 255), 2)

    # Add frame number
    if 'frame_number' in df.columns:
        cv2.putText(frame, f"Frame: {int(row['frame_number'])}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    return frame

# Function to generate the video from pose data
def generate_video(csv_path, output_path, frame_size=(3840, 2160), fps=60, bg_color=(20, 20, 20), joints=None, connections=None):
    """
    Generate a video from the pose data by creating frames and saving them.
    
    Preconditions:
        - `csv_path`: The path to the CSV containing pose data.
        - `output_path`: The path to save the generated video.
        - `frame_size`: The target size of the video (height, width).
        - `fps`: Frames per second of the video.
        - `bg_color`: Background color for the frames.
        - `joints`: A list of joint names to visualize.
        - `connections`: A list of connections between joints for skeleton drawing.
        
    Postconditions:
        - The generated video is saved to `output_path`.
    """
    # Load pose data
    df = load_pose_data(csv_path)
    frame_height, frame_width = frame_size
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Better codec compatibility than 'mp4v'
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Estimate original pose data dimensions
    orig_w, orig_h = estimate_original_dimensions(df)

    # Generate video frames and save the video
    for idx, row in df.iterrows():
        frame = create_frame(row, df, joints, connections, orig_w, orig_h, frame_width, frame_height, bg_color)
        out.write(frame)

    out.release()
    print(f"✅ Animation saved to: {output_path}")
def main():
    # Specify paths for the input CSV and the output video
    csv_path = r'Football_throw_video test copy\output_csv\combined_1_cleaned.csv'
    output_path = r'Videos\trimmed_mean_output\Black_BG test1.mp4'

    # Specify the joints and connections for the skeleton
    joints = ['shoulder_left', 'shoulder_right', 'elbow_left', 'elbow_right', 'wrist_left', 'wrist_right', 'hip_left', 'hip_right', 'knee_left', 'knee_right', 'ankle_left', 'ankle_right']
    connections = [
        ('shoulder_left', 'elbow_left'),
        ('elbow_left', 'wrist_left'),
        ('shoulder_right', 'elbow_right'),
        ('elbow_right', 'wrist_right'),
        ('hip_left', 'knee_left'),
        ('knee_left', 'ankle_left'),
        ('hip_right', 'knee_right'),
        ('knee_right', 'ankle_right'),
        ('shoulder_left', 'shoulder_right'),
        ('hip_left', 'hip_right'),
    ]

    # Call the function to generate the video
    generate_video(csv_path, output_path, frame_size=(3840, 2160), fps=60, bg_color=(20, 20, 20), joints=joints, connections=connections)

if __name__ == '__main__':
    main()
