# Name: Mark William
# ID: 120210348
# LAB 2 Assignment (BONUS)

import cv2
import numpy as np
import time
import os # To check file existence

# --- Configuration ---
QUERY_IMAGE_FILENAME = 'query1.png'
VIDEO_FILENAME = 'input_video.mp4'
OUTPUT_FILENAME = 'output_video_sift.mp4'
SAVE_OUTPUT_VIDEO = True 

MIN_MATCH_COUNT = 10       # Minimum number of good matches for homography
RATIO_THRESH = 0.75      # Lowe's ratio test threshold

# --- Step 1: Load Query Image and Initialize SIFT ---
# Check if query image exists
if not os.path.exists(QUERY_IMAGE_FILENAME):
    print(f"Error: Query image '{QUERY_IMAGE_FILENAME}' not found.")
    exit()

try:
    query_img_bgr = cv2.imread(QUERY_IMAGE_FILENAME)
    if query_img_bgr is None:
        print(f"Error: Could not load query image '{QUERY_IMAGE_FILENAME}'.")
        exit()
    query_img_gray = cv2.cvtColor(query_img_bgr, cv2.COLOR_BGR2GRAY)
    h_query, w_query = query_img_gray.shape[:2]
    print(f"Query image '{QUERY_IMAGE_FILENAME}' loaded ({w_query}x{h_query}).")

except Exception as e:
    print(f"Error loading query image: {e}")
    exit()

# Initialize SIFT Detector and Brute-Force Matcher (with NORM_L2 for SIFT)
print("Using SIFT detector...")
try:
    detector = cv2.SIFT_create()
    # Use NORM_L2 for SIFT descriptors (float)
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
except cv2.error as e:
    print("\nError initializing SIFT. Is 'opencv-contrib-python' installed?")
    print("Install it using: pip install opencv-contrib-python")
    print("(You might need to uninstall opencv-python first: pip uninstall opencv-python)")
    exit()
except Exception as e:
    print(f"Error initializing SIFT: {e}")
    exit()

# --- Step 2: Find Keypoints and Descriptors for the Query Image (Done Once) ---
try:
    kp_query, des_query = detector.detectAndCompute(query_img_gray, None)
    if des_query is None or len(kp_query) == 0:
        print("Error: No descriptors found in the query image. Cannot proceed.")
        exit()
    print(f"{len(kp_query)} SIFT keypoints found in query image.")
    print(f"Query descriptor size: {des_query.shape[1]}") # Should be 128

except Exception as e:
    print(f"Error detecting query features: {e}")
    exit()

# --- Step 3: Initialize Video Capture from File ---

# Check if video file exists
if not os.path.exists(VIDEO_FILENAME):
    print(f"Error: Video file '{VIDEO_FILENAME}' not found.")
    exit()

cap = cv2.VideoCapture(VIDEO_FILENAME)
if not cap.isOpened():
    print(f"Error: Could not open video file '{VIDEO_FILENAME}'.")
    exit()

# Get video properties for saving (if needed)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_video = cap.get(cv2.CAP_PROP_FPS)
print(f"Input video properties: {frame_width}x{frame_height} @ {fps_video:.2f} FPS")

# --- Step 4: Initialize Video Writer ---
video_writer = None
if SAVE_OUTPUT_VIDEO:
    # Define the codec and create VideoWriter object
    # Common codecs: 'mp4v' for .mp4, 'XVID' for .avi
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(OUTPUT_FILENAME, fourcc, fps_video, (frame_width, frame_height))
    if not video_writer.isOpened():
        print(f"Error: Could not open video writer for '{OUTPUT_FILENAME}'. Saving disabled.")
        SAVE_OUTPUT_VIDEO = False
    else:
        print(f"Output video will be saved to '{OUTPUT_FILENAME}'")


# --- Step 5: Process Video Frames ---
print("Starting video processing (using SIFT - may be slow)... Press 'q' to quit.")
frame_number = 0
start_process_time = time.time()

while cap.isOpened():
    ret, frame_bgr = cap.read()
    if not ret:
        print("End of video file reached.")
        break

    frame_number += 1
    frame_time_start = time.time()

    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    frame_display = frame_bgr.copy() # Work on a copy

    # --- Step 5a: Detect Keypoints and Descriptors for the Current Frame ---
    kp_frame, des_frame = detector.detectAndCompute(frame_gray, None)

    # --- Step 5b: Match Descriptors ---
    matches = []
    good_matches = []
    if des_frame is not None and len(kp_frame) > 0:
        # Use knnMatch with k=2 for ratio test
        matches = matcher.knnMatch(des_query, des_frame, k=2)

        # --- Step 5c: Apply Lowe's Ratio Test ---
        # Ensure we got pairs of matches before iterating
        if matches and len(matches[0]) == 2:
             for m, n in matches:
                 if m.distance < RATIO_THRESH * n.distance:
                     good_matches.append(m) # Keep the best match

    # --- Step 5d: Estimate Homography and Draw Bounding Box ---
    object_found = False
    if len(good_matches) >= MIN_MATCH_COUNT:
        # Extract location of good matches
        src_pts = np.float32([ kp_query[m.queryIdx].pt for m in good_matches ]).reshape(-1, 1, 2)
        dst_pts = np.float32([ kp_frame[m.trainIdx].pt for m in good_matches ]).reshape(-1, 1, 2)

        # Find homography (perspective transformation)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) # RANSAC needs min 4 points

        if M is not None:
            object_found = True
            # Apply perspective transform to the corners of the query image
            pts_query_corners = np.float32([ [0, 0], [0, h_query-1],
                                             [w_query-1, h_query-1], [w_query-1, 0] ]).reshape(-1, 1, 2)
            dst_corners = cv2.perspectiveTransform(pts_query_corners, M)

            # Draw the bounding box (polygon) on the frame
            frame_display = cv2.polylines(frame_display, [np.int32(dst_corners)],
                                          True, (0, 255, 0), 3, cv2.LINE_AA) # Green box

    # --- Step 5e: Display Info and Frame ---
    frame_time_end = time.time()
    process_duration = frame_time_end - frame_time_start
    fps_current = 1.0 / process_duration if process_duration > 0 else 0

    info_text = f"Frame: {frame_number} | FPS: {fps_current:.1f} | Matches: {len(good_matches)}"
    status_text = "Detected" if object_found else "Not Detected"
    status_color = (0, 255, 0) if object_found else (0, 0, 255)

    cv2.putText(frame_display, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) # Yellow text
    cv2.putText(frame_display, status_text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    cv2.imshow('Object Detection in Video (SIFT)', frame_display)

    # --- Step 5f: Save Frame---
    if SAVE_OUTPUT_VIDEO and video_writer is not None:
        video_writer.write(frame_display)

    # --- Step 5g: Exit Condition ---
    if cv2.waitKey(1) & 0xFF == ord('q'): # waitKey(1) is important for video display
        print("Exiting...")
        break

# --- Step 6: Release Resources ---
total_time = time.time() - start_process_time
print(f"\nProcessed {frame_number} frames in {total_time:.2f} seconds.")
cap.release()
if SAVE_OUTPUT_VIDEO and video_writer is not None:
    video_writer.release()
    print(f"Output video saved to '{OUTPUT_FILENAME}'")
cv2.destroyAllWindows()
print("Video capture released and windows closed.")