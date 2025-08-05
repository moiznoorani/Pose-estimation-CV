import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import torch
import numpy as np
import os
import glob
import csv
import mediapipe as mp
from ultralytics import YOLO

# Initialize MediaPipe Pose module
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
DEBUG_LOG_PATH = "video_debug_log.txt"

def log_debug(message):
    with open(DEBUG_LOG_PATH, "a") as f:
        f.write(message + "\n")

def calculate_reference_point(landmarks, frame_width, frame_height):
    points = [11, 12, 23, 24]  # Right Shoulder, Left Shoulder, Right Hip, Left Hip
    x = np.mean([landmarks[point].x * frame_width for point in points])
    y = np.mean([landmarks[point].y * frame_height for point in points])
    return (x, y)

def normalize_coordinates(coord, reference_point):
    return (coord[0] - reference_point[0], coord[1] - reference_point[1])

def process_video_with_pose_and_ball(input_path, output_video_path, csv_output_path, model):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {input_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    orientation = "Portrait (9:16)" if frame_height > frame_width else "Landscape (16:9)"
    log_debug(f"Processing {input_path}")
    log_debug(f"Frame size: width={frame_width}, height={frame_height}")
    log_debug(f"Inferred orientation: {orientation}")
    
    rotate_frame = False
    if frame_width > frame_height:
        rotate_frame = True
        frame_width, frame_height = frame_height, frame_width  # Swap for rotated video size
        log_debug("Manual rotation applied: 90 degrees clockwise to portrait")

    log_debug("-" * 50)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    with open(csv_output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "frame_number", "ball_x", "ball_y", "ball_confidence",
            "ref_x", "ref_y"
        ] + [
            f"{joint}_{coord}" for joint in [
                "shoulder_left", "shoulder_right", "elbow_left", "elbow_right",
                "wrist_left", "wrist_right", "hip_left", "hip_right",
                "knee_left", "knee_right", "ankle_left", "ankle_right"
            ] for coord in ["x", "y", "rel_x", "rel_y"]
        ])


        with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
            frame_index = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_index += 1

                if rotate_frame:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                # Ball Detection
                results = model(frame)
                detections = results[0].boxes.xyxy.cpu().numpy()
                ball_center = (None, None)
                best_conf = 0.0

                for *bbox, conf, cls_id in detections:
                    if int(cls_id) == 32:  # class 32 = sports ball
                        x1, y1, x2, y2 = map(int, bbox)
                        c = float(conf)
                        if c > best_conf:
                            best_conf = c
                            ball_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                            # Draw ball bounding box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

                # Pose Detection
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results_pose = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                data_row = [frame_index, ball_center[0], ball_center[1], best_conf]

                if results_pose.pose_landmarks:
                    landmarks = results_pose.pose_landmarks.landmark
                    ref_point = calculate_reference_point(landmarks, frame_width, frame_height)
                    data_row.extend([ref_point[0], ref_point[1]])

                    keypoints = {
                        "shoulder_left": landmarks[11],
                        "shoulder_right": landmarks[12],
                        "elbow_left": landmarks[13],
                        "elbow_right": landmarks[14],
                        "wrist_left": landmarks[15],
                        "wrist_right": landmarks[16],
                        "hip_left": landmarks[23],
                        "hip_right": landmarks[24],
                        "knee_left": landmarks[25],
                        "knee_right": landmarks[26],
                        "ankle_left": landmarks[27],
                        "ankle_right": landmarks[28]
                    }

                    for name, point in keypoints.items():
                        abs_coord = (point.x * frame_width, point.y * frame_height)
                        rel_coord = normalize_coordinates(abs_coord, ref_point)
                        data_row.extend([abs_coord[0], abs_coord[1], rel_coord[0], rel_coord[1]])


                    mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                else:
                    data_row.extend([None]*24)  # Fill in missing pose data

                writer.writerow(data_row)

                out.write(frame)
                cv2.imshow("Pose & Ball Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Saved: {output_video_path}, CSV: {csv_output_path}")

def main():
    model = YOLO('yolov8m.pt')
    model.conf = 0.35
    model.iou = 0.45

    folder = '/Users/moiznoorani/Library/CloudStorage/OneDrive-MNSCU/Pose-estimation-CV/Football_throw_video test'
    video_files = glob.glob(os.path.join(folder, "*.mp4"))

    if not video_files:
        print("No videos found in the folder.")
        return

    output_video_dir = os.path.join(folder, "output_combined")
    output_csv_dir = os.path.join(folder, "output_csv")
    os.makedirs(output_video_dir, exist_ok=True)
    os.makedirs(output_csv_dir, exist_ok=True)

    for idx, input_path in enumerate(sorted(video_files), start=1):
        output_video = os.path.join(output_video_dir, f"combined_{idx}.mp4")
        output_csv = os.path.join(output_csv_dir, f"combined_{idx}.csv")
        process_video_with_pose_and_ball(input_path, output_video, output_csv, model)

    print("\nAll videos processed.")

if __name__ == "__main__":
    main()
