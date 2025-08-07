import os
import glob
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

from Step_1_aruco_scale import process_videos
from Step_2_Pose_detection_data import process_video_with_pose_and_ball
from Step_3_Data_cleanup import process_csv_cleanup

# -------------------------------
# Shift trimmed mean pose
# -------------------------------
def shift_trimmed_mean(trimmed_csv, user_csv):
    df_trimmed = pd.read_csv(trimmed_csv)
    df_user = pd.read_csv(user_csv)

    min_frames = min(len(df_trimmed), len(df_user))
    df_trimmed = df_trimmed.iloc[:min_frames].copy()
    df_user = df_user.iloc[:min_frames].copy()

    joints = ["shoulder_left", "shoulder_right", "elbow_left", "elbow_right",
              "wrist_left", "wrist_right", "hip_left", "hip_right",
              "knee_left", "knee_right", "ankle_left", "ankle_right"]

    for i in range(min_frames):
        dx = df_user.at[i, 'ref_x'] - df_trimmed.at[i, 'ref_x']
        dy = df_user.at[i, 'ref_y'] - df_trimmed.at[i, 'ref_y']

        for joint in joints:
            x_col = f"{joint}_x_cleaned"
            y_col = f"{joint}_y_cleaned"
            if x_col in df_trimmed.columns:
                df_trimmed.at[i, x_col] += dx
            if y_col in df_trimmed.columns:
                df_trimmed.at[i, y_col] += dy

        df_trimmed.at[i, 'ref_x'] += dx
        df_trimmed.at[i, 'ref_y'] += dy

    return df_trimmed

# -------------------------------
# Draw overlay on video
# -------------------------------
def draw_pose(frame, coords, connections, color):
    for joint, (x, y) in coords.items():
        cv2.circle(frame, (x, y), 8, color, -1)
    for j1, j2 in connections:
        if j1 in coords and j2 in coords:
            cv2.line(frame, coords[j1], coords[j2], color, 2)
    return frame


def overlay_dual_pose_video(video_path, df_trimmed, df_user, output_path):
    """
    Creates a side-by-side video:
    - Left: Original video with trimmed mean pose overlay (in red)
    - Right: Original video with user pose overlay (in green)

    Preconditions:
        - video_path: Path to the user's scaled video.
        - df_trimmed: Shifted trimmed mean pose DataFrame.
        - df_user: Cleaned user pose DataFrame.
        - output_path: File path to save the final output video.

    Postconditions:
        - Saves a side-by-side comparison video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Failed to open video.")
        return

    ret, first = cap.read()
    if not ret:
        print("❌ Failed to read first frame.")
        return

    if first.shape[1] > first.shape[0]:
        first = cv2.rotate(first, cv2.ROTATE_90_CLOCKWISE)
    h, w = first.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    out = None
    frame_idx = 0
    total_frames = min(len(df_trimmed), len(df_user), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    joints = ["shoulder_left", "shoulder_right", "elbow_left", "elbow_right",
              "wrist_left", "wrist_right", "hip_left", "hip_right",
              "knee_left", "knee_right", "ankle_left", "ankle_right"]

    connections = [
        ("shoulder_left", "elbow_left"), ("elbow_left", "wrist_left"),
        ("shoulder_right", "elbow_right"), ("elbow_right", "wrist_right"),
        ("hip_left", "knee_left"), ("knee_left", "ankle_left"),
        ("hip_right", "knee_right"), ("knee_right", "ankle_right"),
        ("shoulder_left", "shoulder_right"), ("hip_left", "hip_right"),
        ("shoulder_left", "hip_left"), ("shoulder_right", "hip_right")
    ]

    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame.shape[1] > frame.shape[0]:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Left: overlay trimmed mean
        frame_trimmed = frame.copy()
        row_trimmed = df_trimmed.iloc[frame_idx]
        trimmed_coords = {
            joint: (int(row_trimmed[f"{joint}_x_cleaned"]), int(row_trimmed[f"{joint}_y_cleaned"]))
            for joint in joints
            if not pd.isna(row_trimmed.get(f"{joint}_x_cleaned")) and not pd.isna(row_trimmed.get(f"{joint}_y_cleaned"))
        }
        draw_pose(frame_trimmed, trimmed_coords, connections, (0, 0, 255))  # Red

        # Right: overlay user pose
        frame_user = frame.copy()
        row_user = df_user.iloc[frame_idx]
        user_coords = {
            joint: (int(row_user[f"{joint}_x"]), int(row_user[f"{joint}_y"]))
            for joint in joints
            if not pd.isna(row_user.get(f"{joint}_x")) and not pd.isna(row_user.get(f"{joint}_y"))
        }
        draw_pose(frame_user, user_coords, connections, (0, 255, 0))  # Green

        # Combine both sides
        side_by_side = cv2.hconcat([frame_trimmed, frame_user])

        if out is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            out = cv2.VideoWriter(output_path, fourcc, fps, (w * 2, h))

        out.write(side_by_side)
        frame_idx += 1

    cap.release()
    if out:
        out.release()
    print(f"✅ Dual overlay side-by-side video saved to: {output_path}")


def overlay_on_video(video_path, df_trimmed, df_user, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Failed to open video.")
        return

    ret, first = cap.read()
    if not ret:
        print("❌ Failed to read first frame.")
        return
    if first.shape[1] > first.shape[0]:
        first = cv2.rotate(first, cv2.ROTATE_90_CLOCKWISE)
    h, w = first.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    out = None
    frame_idx = 0
    total_frames = min(len(df_trimmed), len(df_user), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    joints = ["shoulder_left", "shoulder_right", "elbow_left", "elbow_right",
              "wrist_left", "wrist_right", "hip_left", "hip_right",
              "knee_left", "knee_right", "ankle_left", "ankle_right"]

    connections = [
        ("shoulder_left", "elbow_left"), ("elbow_left", "wrist_left"),
        ("shoulder_right", "elbow_right"), ("elbow_right", "wrist_right"),
        ("hip_left", "knee_left"), ("knee_left", "ankle_left"),
        ("hip_right", "knee_right"), ("knee_right", "ankle_right"),
        ("shoulder_left", "shoulder_right"), ("hip_left", "hip_right"),
        ("shoulder_left", "hip_left"), ("shoulder_right", "hip_right")
    ]

    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame.shape[1] > frame.shape[0]:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        row_trimmed = df_trimmed.iloc[frame_idx]
        row_user = df_user.iloc[frame_idx]

        trimmed_coords = {}
        user_coords = {}

        for joint in joints:
            x_trim = f"{joint}_x_cleaned"
            y_trim = f"{joint}_y_cleaned"
            x_user = f"{joint}_x"
            y_user = f"{joint}_y"

            if not pd.isna(row_trimmed.get(x_trim)) and not pd.isna(row_trimmed.get(y_trim)):
                trimmed_coords[joint] = (int(row_trimmed[x_trim]), int(row_trimmed[y_trim]))
            if not pd.isna(row_user.get(x_user)) and not pd.isna(row_user.get(y_user)):
                user_coords[joint] = (int(row_user[x_user]), int(row_user[y_user]))

        draw_pose(frame, trimmed_coords, connections, (0, 0, 255))   # Red: trimmed mean
        draw_pose(frame, user_coords, connections, (0, 255, 0))      # Green: user

        if out is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        out.write(frame)
        frame_idx += 1

    cap.release()
    if out:
        out.release()
    print(f"✅ Final overlay video saved to: {output_path}")

# -------------------------------
# Combined Pipeline Execution
# -------------------------------
if __name__ == "__main__":
    VIDEO_DIR = r"C:\Users\noora\OneDrive - MNSCU (1)\Pose-estimation-CV\Videos\Test"
    SCALED_DIR = os.path.join(VIDEO_DIR, "output_scaled")
    PROCESSED_VIDEO_DIR = os.path.join(VIDEO_DIR, "processed_output_videos")
    PROCESSED_CSV_DIR = os.path.join(VIDEO_DIR, "processed_output_csv")
    CLEANED_CSV_DIR = os.path.join(VIDEO_DIR, "cleaned_csv_output")
    TRIMMED_CSV_PATH = os.path.join(VIDEO_DIR, "trimmed_mean_output", "trimmed_mean_poses.csv")
    OUTPUT_OVERLAY = os.path.join(VIDEO_DIR, "trimmed_mean_output", "test_overlay_final_overlay.mp4")

    # Step 1: Scale
    print("🔧 Step 1: Scaling video...")
    process_videos(VIDEO_DIR, desired_marker_size_px=70)

    scaled_videos = sorted(glob.glob(os.path.join(SCALED_DIR, "*.mov")))
    if not scaled_videos:
        print("❌ No scaled video found.")
        exit()
    scaled_video_path = scaled_videos[0]

    # Step 2: Pose & Ball Detection
    print("🎯 Step 2: Detecting pose + ball...")
    model = YOLO("yolov8m.pt")
    model.conf = 0.35
    model.iou = 0.45
    model.verbose = False

    raw_csv_path = os.path.join(PROCESSED_CSV_DIR, "test_overlay.csv")
    processed_video_path = os.path.join(PROCESSED_VIDEO_DIR, "test_overlay_output.mp4")
    os.makedirs(PROCESSED_CSV_DIR, exist_ok=True)
    os.makedirs(PROCESSED_VIDEO_DIR, exist_ok=True)
    process_video_with_pose_and_ball(scaled_video_path, processed_video_path, raw_csv_path, model)

    # Step 3: CSV Cleanup
    print("🧹 Step 3: Cleaning CSV...")
    process_csv_cleanup(PROCESSED_CSV_DIR, CLEANED_CSV_DIR)
    user_cleaned_csv = os.path.join(CLEANED_CSV_DIR, "test_overlay_cleaned.csv")

    # Step 4: Align Trimmed Mean
    print("📐 Step 4: Shifting trimmed mean...")
    df_shifted = shift_trimmed_mean(TRIMMED_CSV_PATH, user_cleaned_csv)

    # Step 5: Overlay and Output
    print("🎥 Step 5: Drawing overlay...")
    df_user = pd.read_csv(user_cleaned_csv)
    overlay_on_video(scaled_video_path, df_shifted, df_user, OUTPUT_OVERLAY)


    # Step 6: Side-by-side overlay
    print("🖼 Step 6: Drawing side-by-side overlay...")
    output_dual_path = os.path.join(VIDEO_DIR, "trimmed_mean_output", "test_overlay_dual_overlay.mp4")
    overlay_dual_pose_video(scaled_video_path, df_shifted, df_user, output_dual_path)

