import cv2
import pandas as pd
import numpy as np
import os
import re

# Landmark class to store joint coordinates (x, y)
class Landmark:
    def __init__(self, x, y):
        """
        Initialize a Landmark object representing a joint in the pose estimation.

        Parameters:
        - x: The x-coordinate of the joint (normalized or pixel value depending on your data).
        - y: The y-coordinate of the joint (normalized or pixel value depending on your data).
        """
        self.x = x
        self.y = y

# Function to load the pose data (trimmed mean pose)
def load_pose_data(csv_path):
    return pd.read_csv(csv_path)

# Function to normalize joint coordinates based on the frame size
def normalize_point(x, y, original_w, original_h, target_w, target_h, padding=100):
    scale_x = (target_w - 2 * padding) / original_w
    scale_y = (target_h - 2 * padding) / original_h
    x = int(x * scale_x + padding)
    y = int(y * scale_y + padding)
    return x, y 

# Function to calculate the reference point (rel_x, rel_y) using key joints by name
def calculate_reference_point(landmarks, frame_width, frame_height):
    # Use joint names instead of indices for clarity
    points = ["shoulder_right", "shoulder_left", "hip_right", "hip_left"]  # Right Shoulder, Left Shoulder, Right Hip, Left Hip
    x = np.mean([landmarks[point].x * frame_width for point in points if point in landmarks])
    y = np.mean([landmarks[point].y * frame_height for point in points if point in landmarks])
    return (x, y)

# Function to center and scale the pose
# Function to center and scale the pose
def center_and_scale_pose(pose_coords, rel_x, rel_y, orig_w, orig_h, frame_width, frame_height):
    # Normalize pose so that it fits into the frame
    norm_coords = {}
    for joint, (x, y) in pose_coords.items():
        norm_coords[joint] = normalize_point(x, y, orig_w, orig_h, frame_width, frame_height)

    # Calculate the center of the pose using shoulder or hip joints (or use any other strategy to find the center)
    pose_center = np.mean([norm_coords[joint] for joint in ['shoulder_left', 'shoulder_right', 'hip_left', 'hip_right']], axis=0)

    # Get the center of the frame
    center_x, center_y = frame_width // 2, frame_height // 2

    # Calculate the offset to center the pose correctly
    offset_x = center_x - pose_center[0]
    offset_y = center_y - pose_center[1]

    # Apply the offset to center the pose
    for joint in norm_coords:
        norm_coords[joint] = (norm_coords[joint][0] + offset_x, norm_coords[joint][1] + offset_y)

    return norm_coords


# Function to draw the pose on the frame
def overlay_pose_on_frame(frame, pose_coords, connections, color):
    # Draw joints
    for joint, (x, y) in pose_coords.items():
        cv2.circle(frame, (x, y), 8, color, -1)
        cv2.putText(frame, joint, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Draw skeleton lines
    for pt1, pt2 in connections:
        if pt1 in pose_coords and pt2 in pose_coords:
            cv2.line(frame, pose_coords[pt1], pose_coords[pt2], color, 2)

    return frame

# Function to generate the video with overlaid trimmed mean pose and original pose
def overlay_points_side_by_side(video_path, csv_path, output_path, point_radius=8):
    df = pd.read_csv(csv_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    out = None
    frame_idx = 0

    joints = [
        "ref", "shoulder_left", "shoulder_right", "elbow_left", "elbow_right",
        "wrist_left", "wrist_right", "hip_left", "hip_right",
        "knee_left", "knee_right", "ankle_left", "ankle_right"
    ]

    connections = [
        ("shoulder_left", "elbow_left"), ("elbow_left", "wrist_left"),
        ("shoulder_right", "elbow_right"), ("elbow_right", "wrist_right"),
        ("hip_left", "knee_left"), ("knee_left", "ankle_left"),
        ("hip_right", "knee_right"), ("knee_right", "ankle_right"),
        ("shoulder_left", "shoulder_right"), ("hip_left", "hip_right"),
        ("shoulder_left", "hip_left"), ("shoulder_right", "hip_right")
    ]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= len(df):
            break

        # Rotate video counterclockwise (90 degrees)
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        h, w = frame.shape[:2]
        combined = np.zeros((h, w * 2, 3), dtype=np.uint8)

        left_frame = frame.copy()
        right_frame = frame.copy()
        row = df.iloc[frame_idx]

        # Left: trimmed mean keypoints (red)
        joint_coords_trimmed = {}
        for joint in joints:
            x_col = f"{joint}_x_cleaned"
            y_col = f"{joint}_y_cleaned"
            if x_col in df.columns and y_col in df.columns:
                try:
                    x, y = int(float(row[x_col])), int(float(row[y_col]))
                    if x > 0 and y > 0:
                        joint_coords_trimmed[joint] = (x, y)
                        cv2.circle(left_frame, (x, y), point_radius, (0, 0, 255), -1)
                        cv2.putText(left_frame, joint, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                except:
                    continue
        for pt1, pt2 in connections:
            if pt1 in joint_coords_trimmed and pt2 in joint_coords_trimmed:
                cv2.line(left_frame, joint_coords_trimmed[pt1], joint_coords_trimmed[pt2], (0, 0, 255), 2)

        # Right: original keypoints (green)
        joint_coords_original = {}
        for joint in joints:
            x_col = f"{joint}_x"
            y_col = f"{joint}_y"
            if x_col in df.columns and y_col in df.columns:
                try:
                    x, y = int(float(row[x_col])), int(float(row[y_col]))
                    if x > 0 and y > 0:
                        joint_coords_original[joint] = (x, y)
                        cv2.circle(right_frame, (x, y), point_radius, (0, 255, 0), -1)
                        cv2.putText(right_frame, joint, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                except:
                    continue
        for pt1, pt2 in connections:
            if pt1 in joint_coords_original and pt2 in joint_coords_original:
                cv2.line(right_frame, joint_coords_original[pt1], joint_coords_original[pt2], (0, 255, 0), 2)

        # Combine side-by-side
        combined[:, :w] = left_frame
        combined[:, w:] = right_frame

        # Write to output
        if out is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            out = cv2.VideoWriter(output_path, fourcc, fps, (w * 2, h))

        out.write(combined)
        frame_idx += 1

    cap.release()
    if out:
        out.release()

    print(f"✅ Side-by-side overlay saved to: {output_path}")

# Example usage:
video_path = "Football_throw_video test\part_1.mp4"
csv_path = "Videos\trimmed_mean_output\trimmed_mean_poses.csv"
output_path = "Videos\trimmed_mean_output\SBS_overlay.mp4"

overlay_points_side_by_side(video_path, csv_path, output_path)
