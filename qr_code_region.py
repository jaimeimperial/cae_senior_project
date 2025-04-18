import cv2
import mediapipe as mp
import numpy as np
from scipy import stats
from collections import deque
import time

# Define color range for light detection (e.g., red light)
color = "test_yellow"
lower_color = np.array(colors[color][0])
upper_color = np.array(colors[color][1])

hand_mask = np.zeros((1080, 1920), dtype=np.uint8)  # Update if frame size changes

previous_thumb_y = None
previous_thumb_x = None
previous_y = None
previous_x = None
projected_image_box = None
qr_scanned = []
saved_rvec = []
saved_tvec = []
saved_bbox = []
saved_data = []
distance = False

zone_def = QR_ZONE_DEFINITIONS.get("1")
switch_centers = zone_def["centers"]

last_flip_time = [0] * len(switch_centers)  # One timestamp per switch
debounce_delay = 0.8  # seconds
switch_states = ["OFF"] * len(switch_centers)        # Confirmed states
predicted_states = ["OFF"] * len(switch_centers)     # Prediction from hand tracking
pending_flips = [False] * len(switch_centers)        # Waiting for confirmation

# Track thumb movement to detect switch flip
            if len(qr.BOXES) != 0:
                print(qr.BOXES)
                for projected_image_box in qr.BOXES:
                    for i, box in enumerate(projected_image_box):
                        if current_time - last_flip_time[i] > debounce_delay:
                            if (previous_thumb_y is not None) and (previous_thumb_x is not None):
                                if (i <= 2):
                                    t_movement = thumb_tip_y - previous_thumb_y
                                    if abs(t_movement) > 10:  # Threshold for detecting significant movement
                                        if in_box(box, thumb_tip_x, thumb_tip_y) and in_box(box, index_tip_x, index_tip_y):
                                            predicted_state = "ON" if t_movement < 0 else "OFF"  # Flip logic
                                            if predicted_states[i] != predicted_state:
                                                predicted_states[i] = predicted_state
                                                pending_flips[i] = True
                                                last_flip_time[i] = current_time
                                                print(f"Switch {i+1} predicted to {predicted_state}")
                                if (i > 2):
                                    t_movement = thumb_tip_x - previous_thumb_x
                                    if abs(t_movement) > 10:  # Threshold for detecting significant movement
                                        if in_box(box, thumb_tip_x, thumb_tip_y) and in_box(box, index_tip_x, index_tip_y):
                                            predicted_state = "ON" if t_movement < 0 else "OFF"  # Flip logic
                                            if predicted_states[i] != predicted_state:
                                                predicted_states[i] = predicted_state
                                                pending_flips[i] = True
                                                last_flip_time[i] = current_time
                                                print(f"Switch {i+1} predicted to {predicted_state}")

# Main video loop
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    camera_matrix = get_camera_matrix(frame.shape)
    dist_coeffs = np.zeros((4, 1))

    init_qr(frame)
    
    for i in range(len(qr_scanned)):
        # Apply pose transformation to each local center
        zone_def = QR_ZONE_DEFINITIONS.get(saved_data[i])
        switch_centers_np = np.array(zone_def["centers"], dtype=np.float32)  # shape (N, 3)
        transformed_switches = transform_points(switch_centers_np, saved_rvec, saved_tvec)
        # Project transformed 3D points to 2D
        image_points, _ = cv2.projectPoints(transformed_switches.reshape(-1, 1, 3), np.zeros((3, 1)), np.zeros((3, 1)),
                                            get_camera_matrix(frame.shape), np.zeros((4, 1)))
        
        if saved_data[i] == "1":
            projected_switch_centers = [pt.ravel().astype(int) for pt in image_points]
            for i, point in enumerate(image_points):
                x, y = point.ravel().astype(int)
                cv2.circle(frame, (x, y), 6, (0, 255, 255), -1)
                cv2.putText(frame, f"SW{i+1}", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if saved_data[i] == "2":
            projected_button_centers = [pt.ravel().astype(int) for pt in image_points]
            for i, point in enumerate(image_points):
                x, y = point.ravel().astype(int)
                cv2.circle(frame, (x, y), 6, (0, 255, 255), -1)
                cv2.putText(frame, f"BUTTON{i+1}", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    track_color = hand_track_region(frame)
    
    if track_color:
        color_detect_yellow(frame)
    
    cv2.imshow("Multi-QR Zone Detection", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        qr_scanned = []
        saved_rvec = []
        saved_tvec = []
        saved_bbox = []
        saved_data = []

cap.release()
cv2.destroyAllWindows()
