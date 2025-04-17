# Name: Mark William
# ID: 120210348
# LAB 2 Assignment

import cv2
import matplotlib.pyplot as plt
import numpy as np

# --- Step 1: Load Images ---
try:
    query_img_bgr = cv2.imread('query.jpg') # Loads in BGR format
    target_img_bgr = cv2.imread('target.jpg') # Loads in BGR format

    # Check if images loaded successfully
    if query_img_bgr is None:
        print("Error: Could not load query image.")
        exit()
    if target_img_bgr is None:
        print("Error: Could not load target image.")
        exit()

    print("Query image shape:", query_img_bgr.shape)
    print("Target image shape:", target_img_bgr.shape)

    # Convert images to grayscale for SIFT detection
    query_img_gray = cv2.cvtColor(query_img_bgr, cv2.COLOR_BGR2GRAY)
    target_img_gray = cv2.cvtColor(target_img_bgr, cv2.COLOR_BGR2GRAY)

except FileNotFoundError as e:
    print(f"Error: {e}. Make sure 'query.jpg' and 'target.jpg' are in the correct directory.")
    exit()
except Exception as e:
    print(f"An error occurred during image loading: {e}")
    exit()


# --- Step 2: Initialize SIFT Detector ---
# Create a SIFT object. It will detect keypoints and compute descriptors.
try:
    sift = cv2.SIFT_create()
except cv2.error as e:
     print("Error initializing SIFT. Your OpenCV installation might not include SIFT.")
     print("You might need to install 'opencv-contrib-python' if you haven't already:")
     print("pip uninstall opencv-python")
     print("pip install opencv-contrib-python")
     exit()


# --- Step 3: Detect Keypoints and Compute Descriptors ---
# Find keypoints and compute descriptors for both images
kp1, des1 = sift.detectAndCompute(query_img_gray, None)
kp2, des2 = sift.detectAndCompute(target_img_gray, None)

print(f"Keypoints found in query image: {len(kp1)}")
print(f"Keypoints found in target image: {len(kp2)}")

# Check if descriptors were found
if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
    print("Error: No descriptors found in one or both images. Cannot proceed with matching.")
    exit()

print(f"Descriptor size: {des1.shape[1]}") # Should be 128 for SIFT


# --- Step 4: Feature Matching ---
# Use a Brute-Force Matcher with L2 norm (suitable for SIFT)
# We use knnMatch to get the top 2 matches for Lowe's ratio test
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False) # crossCheck=False for knnMatch
matches = bf.knnMatch(des1, des2, k=2)
print(f"Number of initial matches (k=2): {len(matches)}")


# --- Step 5: Apply Lowe's Ratio Test ---
# Filter matches based on the ratio of distances between the best and second-best match
good_matches = []
ratio_thresh = 0.75 # Common threshold value
for m, n in matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m) # Keep the best match 'm'

print(f"Number of good matches after ratio test: {len(good_matches)}")


# --- Step 6: Visualize Good Matches ---
img_matches = cv2.drawMatches(query_img_bgr, kp1, target_img_bgr, kp2,
                              good_matches, None, # Draw only good matches
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


# --- Step 7: Display the Results ---
plt.figure(figsize=(15, 10))
# Convert BGR (OpenCV default) to RGB (Matplotlib default) for display
plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
plt.title(f'Good SIFT Matches (Ratio Test threshold={ratio_thresh})')
plt.axis('off') # Hide axes
plt.show()

# --- Step 8: Draw keypoints on the target image only ---
if len(good_matches) > 0:
    # Get the keypoints from the good matches in the target image (kp2)
    target_kp_indices = [m.trainIdx for m in good_matches]
    target_matched_kps = [kp2[i] for i in target_kp_indices]

    # Draw these keypoints on the target image
    img_target_kps = cv2.drawKeypoints(target_img_bgr, target_matched_kps, None,
                                       color=(0, 255, 0), # Draw in green
                                       flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Display the target image with matched keypoints
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img_target_kps, cv2.COLOR_BGR2RGB))
    plt.title('Matched Keypoints on Target Image')
    plt.axis('off')
    plt.show()
else:
    print("No good matches found to draw on the target image.")