import os
import glob
from Step_1_aruco_scale import process_videos
from Step_2_Pose_detection_data import process_video_with_pose_and_ball
from ultralytics import YOLO
from Step_3_Data_cleanup import process_csv_cleanup
from Step_4_trimmed_mean import trim_mean_poses  # Import the trimmed mean function
from Step_5_generate_video_on_BG import generate_video  # Import the video generation function

# --- CONFIGURATION ---
VIDEO_DIR = "C:\Users\noora\OneDrive - MNSCU (1)\Pose-estimation-CV\Videos"
DESIRED_MARKER_SIZE_PX = 70  # Desired marker size in pixels

# Folder containing the scaled video files (output from aruco_scale)
SCALED_VIDEO_FOLDER = os.path.join(VIDEO_DIR, "output_scaled")

# Output folder for the processed videos and CSVs
OUTPUT_VIDEO_DIR = os.path.join(VIDEO_DIR, "processed_output_videos")
OUTPUT_CSV_DIR = os.path.join(VIDEO_DIR, "processed_output_csv")
CLEANED_CSV_DIR = os.path.join(VIDEO_DIR, "cleaned_csv_output")
TRIMMED_MEAN_CSV_DIR = os.path.join(VIDEO_DIR, "trimmed_mean_output")
GENERATED_VIDEO_DIR = os.path.join(VIDEO_DIR, "generated_video_output")

os.makedirs(SCALED_VIDEO_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)
os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)
os.makedirs(CLEANED_CSV_DIR, exist_ok=True)
os.makedirs(TRIMMED_MEAN_CSV_DIR, exist_ok=True)
os.makedirs(GENERATED_VIDEO_DIR, exist_ok=True)

# Function to process all the videos using the Pose and Ball detection method
def process_all_videos_with_pose_and_ball(input_folder, output_video_dir, output_csv_dir):
    model = YOLO('yolov8m.pt')  # Load the YOLO model
    model.conf = 0.35  # Set the confidence threshold
    model.iou = 0.45   # Set the IOU threshold

    video_files = glob.glob(os.path.join(input_folder, "*.mov"))

    if not video_files:
        print("No videos found in the folder.")
        return

    for idx, input_path in enumerate(sorted(video_files), start=1):
        # Define the output video and CSV paths
        output_video = os.path.join(output_video_dir, f"processed_{idx}.mov")
        output_csv = os.path.join(output_csv_dir, f"processed_{idx}.csv")

        # Call the function to process each video and ball detection
        process_video_with_pose_and_ball(input_path, output_video, output_csv, model)

    print("\nAll videos processed and saved.")

# Run the process for scaling videos and then apply pose & ball detection
if __name__ == "__main__":
    # Step 1: Process videos for scaling (ArUco marker resizing)
    process_videos(VIDEO_DIR, DESIRED_MARKER_SIZE_PX)

    # Step 2: Process the scaled videos with Pose and Ball detection
    process_all_videos_with_pose_and_ball(SCALED_VIDEO_FOLDER, OUTPUT_VIDEO_DIR, OUTPUT_CSV_DIR)

    # Step 3: Clean up the CSV data
    process_csv_cleanup(OUTPUT_CSV_DIR, CLEANED_CSV_DIR)

    # Step 4: Compute trimmed mean poses
    trimmed_mean_csv_path = os.path.join(TRIMMED_MEAN_CSV_DIR, "trimmed_mean_poses.csv")
    trim_mean_poses(CLEANED_CSV_DIR, trimmed_mean_csv_path)

    # Step 5: Generate a video from the trimmed mean poses
    joints = [
    "shoulder_left", "shoulder_right", "elbow_left", "elbow_right",
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
    generated_video_path = os.path.join(GENERATED_VIDEO_DIR, "poses.mp4")
    generate_video(trimmed_mean_csv_path, generated_video_path, frame_size=(3840, 2160), fps=60, bg_color=(20, 20, 20), joints=joints,connections=connections)

    print("All steps completed successfully.")
