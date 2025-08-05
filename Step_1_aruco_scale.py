import os
import cv2
import numpy as np
import glob

# --- CONFIGURATION ---
VIDEO_DIR = "C:\Users\noora\OneDrive - MNSCU (1)\Pose-estimation-CV\Videos"
MARKER_ID = 0
MARKER_SIZE_MM = 50  # Printed marker size in mm


def detect_aruco_marker(frame):
    """
    Detects an ArUco marker in the given frame.

    Preconditions:
        - `frame` is a valid image in the form of a NumPy array (e.g., a frame from a video capture).
        - The image must be in BGR color format.

    Postconditions:
        - Returns the corner coordinates of the detected marker if found, else returns None.
        - The output is a 4x2 NumPy array representing the corner coordinates of the detected marker.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id == MARKER_ID:
                return corners[i].reshape((4, 2))
    return None


def get_marker_pixel_size(marker_corners):
    """
    Calculates the average pixel size of the ArUco marker from its corners.

    Preconditions:
        - `marker_corners` is a 4x2 NumPy array containing the pixel coordinates of the marker's corners.

    Postconditions:
        - Returns the average pixel size (in pixels) of the marker, calculated from its corners.
    """
    tl, tr, br, bl = marker_corners
    top = np.linalg.norm(tr - tl)
    right = np.linalg.norm(br - tr)
    bottom = np.linalg.norm(bl - br)
    left = np.linalg.norm(tl - bl)
    return (top + right + bottom + left) / 4


def save_debug_marker_frame(frame, marker_corners, marker_px_size, video_name, scaled=False):
    """
    Saves a debug image of the frame with the detected ArUco marker and its size displayed.

    Preconditions:
        - `frame` is a valid image (NumPy array).
        - `marker_corners` is a 4x2 NumPy array with the coordinates of the detected marker.
        - `marker_px_size` is a float representing the size of the marker in pixels.
        - `video_name` is the name of the video file being processed.
        - `scaled` is a boolean indicating whether the marker is scaled to a fixed size.

    Postconditions:
        - A debug image is saved showing the detected marker and its size at the given location.
        - A file path to the saved debug image is printed.
    """
    debug_frame = frame.copy()
    corners = marker_corners.astype(int)
    cv2.polylines(debug_frame, [corners], isClosed=True, color=(0, 255, 0), thickness=2)

    tl, tr = corners[0], corners[1]
    center = np.mean([tl, tr], axis=0).astype(int)
    angle = np.degrees(np.arctan2(tr[1] - tl[1], tr[0] - tl[0]))

    label = f"ID {MARKER_ID}\nSize: {marker_px_size:.1f}px"
    if scaled:
        label = f"[Scaled]\n{label}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color = (0, 0, 255)

    text_size, _ = cv2.getTextSize("Size: 000.0px", font, font_scale, thickness)
    text_w, text_h = text_size
    text_img = np.zeros((text_h*4, text_w*2, 3), dtype=np.uint8)
    y0 = 30
    for i, line in enumerate(label.splitlines()):
        y = y0 + i * (text_h + 5)
        cv2.putText(text_img, line, (5, y), font, font_scale, color, thickness)

    M = cv2.getRotationMatrix2D((text_img.shape[1]//2, text_img.shape[0]//2), angle, 1.0)
    rotated = cv2.warpAffine(text_img, M, (text_img.shape[1], text_img.shape[0]), flags=cv2.INTER_LINEAR)

    x, y = center[0], center[1]
    h, w = rotated.shape[:2]
    x1, y1 = max(0, x - w // 2), max(0, y - h // 2)
    x2, y2 = min(debug_frame.shape[1], x1 + w), min(debug_frame.shape[0], y1 + h)
    roi = debug_frame[y1:y2, x1:x2]

    if roi.shape[:2] == rotated.shape[:2]:
        mask = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
        fg = cv2.bitwise_and(rotated, rotated, mask=mask)
        debug_frame[y1:y2, x1:x2] = cv2.add(bg, fg)

    suffix = "scaled" if scaled else "baseline"
    debug_folder = os.path.join(VIDEO_DIR, "Debug_photo")
    os.makedirs(debug_folder, exist_ok=True)

    debug_path = os.path.join(debug_folder, f"debug_marker_frame_{suffix}_{os.path.splitext(os.path.basename(video_name))[0]}.jpg")
    cv2.imwrite(debug_path, debug_frame)
    print(f"🖼️ Saved debug image: {debug_path}")


def resize_video_fixed_resolution(video_path, desired_marker_size_px, output_path):
    """
    Resizes the video frames so that the ArUco marker appears with a fixed size using the scaling factor
    calculated from the first frame.

    Preconditions:
        - `video_path` is the path to the input video file.
        - `desired_marker_size_px` is the desired marker size in pixels.
        - `output_path` is the path to save the scaled video.

    Postconditions:
        - The video is resized such that the ArUco marker appears with the desired pixel size.
        - The resized video is saved at `output_path`.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w, out_h = in_w, in_h

    # Initialize the VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    # Process the first frame to get the scaling factor
    ret, frame = cap.read()
    if not ret:
        print("❌ Couldn't read the first frame.")
        return

    # Detect the ArUco marker in the first frame
    marker_corners = detect_aruco_marker(frame)
    if marker_corners is None:
        print("❌ Marker not found in the first frame.")
        return

    # Get the pixel size of the marker in the first frame
    marker_px = get_marker_pixel_size(marker_corners)
    print(f"📏 Detected marker size in first frame: {marker_px:.2f} pixels")

    # Calculate the scaling factor to make the marker the desired size
    scale = desired_marker_size_px / marker_px
    print(f"🔄 Scaling factor for the video: {scale:.4f}")

    # Apply the same scaling factor to all frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to the first frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame using the calculated scaling factor
        new_w = int(in_w * scale)
        new_h = int(in_h * scale)
        resized = cv2.resize(frame, (new_w, new_h))

        # If the resized frame is smaller than the output resolution, center it on a black background
        if scale < 1:
            result = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            x_offset = (out_w - new_w) // 2
            y_offset = (out_h - new_h) // 2
            result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        else:
            x_start = (new_w - out_w) // 2
            y_start = (new_h - out_h) // 2
            result = resized[y_start:y_start+out_h, x_start:x_start+out_w]

        # Write the resized frame to the output video
        out.write(result)

    cap.release()
    out.release()
    print(f"✅ Video saved to: {output_path}")


def process_videos(video_dir, desired_marker_size_px):
    """
    Processes all video files in the specified directory, scales them to the desired marker size, and saves the results.
    
    Preconditions:
        - `video_dir` is the directory containing video files in supported formats (e.g., mp4, mov).
        - `desired_marker_size_px` is the desired marker size in pixels for scaling.
    
    Postconditions:
        - All videos in the specified directory are processed to have the marker at the desired size.
        - Debug images are saved showing marker detection and scaling.
    """
    # Get all video files (mp4, mov, MP4, MOV)
    video_files = sorted(glob.glob(os.path.join(video_dir, "*.mp4")) +
                         glob.glob(os.path.join(video_dir, "*.mov")) +
                         glob.glob(os.path.join(video_dir, "*.MP4")) +
                         glob.glob(os.path.join(video_dir, "*.MOV")))

    print(f"Found {len(video_files)} video(s)")

    for idx, video_path in enumerate(video_files):
        print(f"\nProcessing {video_path}...")

        # Open the video file and read the 60th frame
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
        ret, frame = cap.read()

        if not ret:
            print("❌ Couldn't read frame 60. Skipping...")
            continue

        # Detect the ArUco marker in the frame
        marker_corners = detect_aruco_marker(frame)
        if marker_corners is None:
            print("❌ Marker not found. Skipping...")
            continue

        # Get the pixel size of the detected marker
        marker_px = get_marker_pixel_size(marker_corners)
        print(f"📏 Detected marker size: {marker_px:.2f} pixels")

        # Save a debug image with the marker details (baseline for first video)
        if idx == 0:
            save_debug_marker_frame(frame, marker_corners, marker_px, video_path, scaled=False)
            print(f"✅ Set as baseline marker (ID {MARKER_ID}, {marker_px:.2f}px)")

        # Calculate scaling factor based on the baseline marker size
        scale = desired_marker_size_px / marker_px
        print(f"🔄 Scaling factor for video: {scale:.4f}")

        # Set the output file path
        SCALED_VIDEO_FOLDER = os.path.join(video_dir, "output_scaled")
        output_file = os.path.join(SCALED_VIDEO_FOLDER ,f"scaled_{os.path.basename(video_path)}")
        

        # Resize the video based on the scale
        resize_video_fixed_resolution(video_path, desired_marker_size_px, output_file)
        print(f"✅ Saved scaled video to: {output_file}")

        # Save debug image showing the scaled marker size
        save_debug_marker_frame(frame, marker_corners, desired_marker_size_px, video_path, scaled=True)
        print(f"✅ Final scaled marker size: {desired_marker_size_px:.2f} px")
