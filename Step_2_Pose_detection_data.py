# ------------------------------
# File: Step_2_pose_detection_data.py
# ------------------------------
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

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
DEBUG_LOG_PATH = "video_debug_log.txt"

def log_debug(message):
    """
    Logs a debug message to the debug log file.

    Preconditions:
        - message: str, message to log

    Postconditions:
        - Appends the message to the debug log file.
    """
    with open(DEBUG_LOG_PATH, "a") as f:
        f.write(message + "\n")

def calculate_reference_point(landmarks, frame_width, frame_height):
    """
    Calculates the average x and y coordinates of shoulders and hips.

    Preconditions:
        - landmarks: list of pose landmarks
        - frame_width, frame_height: dimensions of the video frame

    Postconditions:
        - Returns a tuple (x, y) representing the average reference point
    """
    points = [11, 12, 23, 24]
    x = np.mean([landmarks[pt].x * frame_width for pt in points])
    y = np.mean([landmarks[pt].y * frame_height for pt in points])
    return (x, y)

def normalize_coordinates(coord, reference_point):
    """
    Converts absolute coordinates into coordinates relative to a reference point.

    Preconditions:
        - coord: tuple of (x, y)
        - reference_point: tuple of (x, y)

    Postconditions:
        - Returns a tuple (x_rel, y_rel)
    """
    return (coord[0] - reference_point[0], coord[1] - reference_point[1])

def process_video_with_pose_and_ball(input_path, output_video_path, csv_output_path, model):
    """
    Processes a video to detect pose landmarks and a sports ball per frame.

    Preconditions:
        - input_path: path to input video
        - output_video_path: path to save annotated video
        - csv_output_path: path to save CSV with pose data
        - model: YOLOv8 object detection model

    Postconditions:
        - Saves an annotated video and a CSV with pose + ball data.
    """
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
        frame_width, frame_height = frame_height, frame_width
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
                    if int(cls_id) == 32:
                        x1, y1, x2, y2 = map(int, bbox)
                        c = float(conf)
                        if c > best_conf:
                            best_conf = c
                            ball_center = ((x1 + x2) // 2, (y1 + y2) // 2)
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
                    data_row.extend([None]*26)

                writer.writerow(data_row)
                out.write(frame)
                cv2.imshow("Pose & Ball Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Saved: {output_video_path}, CSV: {csv_output_path}")
