import cv2
import mediapipe as mp
import numpy as np
from scipy import stats
from collections import deque
import time

QR_SIZE_CM = 3               # Size of QR code (cm)
FOCAL_LENGTH = 1238.40

colors = {
    'test_purple' :    [[110, 50, 130],    [130, 80, 190]],
    'test_yellow' :    [[20, 60, 60],    [30, 255, 255]],
    'black':    [[180, 255, 30],    [0, 0, 0]],
    'white':    [[180, 18, 255],    [0, 0, 231]],
    'red1':     [[180, 255, 255],   [159, 50, 70]],
    'red2':     [[9, 255, 255],     [0, 50, 70]],
    'green':    [[89, 255, 255],    [36, 50, 70]],
    'blue':     [[128, 255, 255],   [90, 50, 70]],
    'yellow':   [[35, 255, 255],    [25, 50, 70]],
    'purple':   [[158, 255, 255],   [129, 50, 70]],
    'orange':   [[24, 255, 255],    [10, 50, 70]],
    'gray':     [[180, 18, 230],    [0, 0, 40]]
}

# Define color range for light detection (e.g., red light)
color = "test_yellow"
lower_color = np.array(colors[color][0])
upper_color = np.array(colors[color][1])

hand_mask = np.zeros((1080, 1920), dtype=np.uint8)  # Update if frame size changes

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

previous_thumb_y = None
previous_thumb_x = None
previous_y = None
previous_x = None
projected_image_box = None
qr_scanned = False
saved_rvec = None
saved_tvec = None
saved_bbox = None
saved_data = None
distance = False

# Switch Regions
z_offset = 0.5  # Dynamic Z-distance in front of QR

switch_centers = [
        [1.3, 6.3, z_offset],
        [3.5, 6.3, z_offset],
        [6.5, 6.3, z_offset],
        [2.0, 9.7, z_offset],
        [6.2, 9.7, z_offset],
    ]
switch_size = (2.5, 3.0)  # Width and height

last_flip_time = [0] * len(switch_centers)  # One timestamp per switch
debounce_delay = 0.8  # seconds
switch_states = ["OFF"] * len(switch_centers)        # Confirmed states
predicted_states = ["OFF"] * len(switch_centers)     # Prediction from hand tracking
pending_flips = [False] * len(switch_centers)        # Waiting for confirmation


led_boxes_local = []

w, h = switch_size
half_w, half_h = w / 2, h / 2
box_local = np.array([
    [-half_w, -half_h, 0],
    [ half_w, -half_h, 0],
    [ half_w,  half_h, 0],
    [-half_w,  half_h, 0]
], dtype=np.float32)

for cx, cy, cz in switch_centers:
    box = box_local + np.array([cx, cy, cz], dtype=np.float32)
    led_boxes_local.append(box)

""""
w, h = switch_size
half_w, half_h = w / 2, h / 2

for cx, cy, cz in switch_centers:
    box = np.array([
        [cx - half_w, cy - half_h, cz],
        [cx + half_w, cy - half_h, cz],
        [cx + half_w, cy + half_h, cz],
        [cx - half_w, cy + half_h, cz]
    ], dtype=np.float32)
    led_boxes_local.append(box)
"""

N = 5
depth_history = deque(maxlen=N)
yaw_history = deque(maxlen=N)
pitch_history = deque(maxlen=N)
roll_history = deque(maxlen=N)

object_points = np.array([
    [0, 0, 0],
    [QR_SIZE_CM, 0, 0],
    [QR_SIZE_CM, QR_SIZE_CM, 0],
    [0, QR_SIZE_CM, 0]
], dtype=np.float32)

def in_box(box, x, y):
    right_edge = max(box[1][0], box[2][0])
    top_edge = max(box[2][1], box[3][1])
    left_edge = min(box[0][0], box[3][0])
    bottom_edge = min(box[0][1], box[1][1])
    if (bottom_edge <= y <= top_edge) and (left_edge <= x <= right_edge):
        return True
    else:
        return False

def detect_single_qr_code(frame):
    qr_decoder = cv2.QRCodeDetector()
    data, bbox, _ = qr_decoder.detectAndDecode(frame)
    if bbox is not None and data:
        return data, bbox[0].astype(np.float32)
    return None, None

def get_camera_matrix(frame_shape, focal_length=FOCAL_LENGTH):
    cx = frame_shape[1] / 2
    cy = frame_shape[0] / 2
    return np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float64)

def draw_qr_pose_info(frame, bbox, data, rvec, tvec, camera_matrix, dist_coeffs):
    # Draw bounding box
    for i in range(len(bbox)):
        pt1 = tuple(bbox[i].astype(int))
        pt2 = tuple(bbox[(i + 1) % len(bbox)].astype(int))
        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    center = np.mean(bbox, axis=0).astype(int)

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

    yaw, pitch, roll = angles[1], angles[0], angles[2]
    z = tvec[2][0]  # Depth in cm

    # Smoothing
    depth_history.append(z)
    yaw_history.append(yaw)
    pitch_history.append(pitch)
    roll_history.append(roll)

    avg_depth = sum(depth_history) / len(depth_history)
    avg_yaw = sum(yaw_history) / len(yaw_history)
    avg_pitch = sum(pitch_history) / len(pitch_history)
    avg_roll = sum(roll_history) / len(roll_history)

    # Draw center
    cv2.circle(frame, tuple(center), 5, (0, 0, 255), -1)

    # Overlay smoothed info
    cv2.putText(frame, f"{data}", (int(bbox[0][0]), int(bbox[0][1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, f"Depth: {avg_depth:.2f} cm", (center[0] + 10, center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(frame, f"Yaw: {avg_yaw:.2f}°", (center[0] + 10, center[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(frame, f"Pitch: {avg_pitch:.2f}°", (center[0] + 10, center[1] + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(frame, f"Roll: {avg_roll:.2f}°", (center[0] + 10, center[1] + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Draw each switch region
    global projected_image_box 
    projected_image_box = []

    for i, box in enumerate(led_boxes_local):
        image_box, _ = cv2.projectPoints(box, rvec, tvec, camera_matrix, dist_coeffs)
        image_box = image_box.reshape(-1, 2).astype(int)

        projected_image_box.append(image_box)

        cv2.polylines(frame, [image_box], isClosed=True, color=(255, 255, 0), thickness=2)
        cx, cy = np.mean(image_box, axis=0).astype(int)
        cv2.putText(frame, f"LED {i+1}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    """
    print(f"QR Code: {data}")
    print(f"Position: x={tvec[0][0]:.2f} cm, y={tvec[1][0]:.2f} cm, z={avg_depth:.2f} cm")
    print(f"Orientation: Yaw={avg_yaw:.2f}°, Pitch={avg_pitch:.2f}°, Roll={avg_roll:.2f}°")
    for i, point in enumerate(bbox):
        print(f"Corner {i + 1}: (X: {point[0]:.1f}, Y: {point[1]:.1f})")
    print("-" * 40)
    """

def transform_points(points, rvec, tvec):
    # Convert rvec to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    transformed = []
    for pt in points:
        world_pt = R @ pt.reshape(3, 1) + tvec
        transformed.append(world_pt.flatten())
    return np.array(transformed, dtype=np.float32)

def hand_track_region(frame):
    global previous_thumb_x, previous_thumb_y, distance
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            global hand_mask
            hand_mask[:] = 0  # Clear the mask

            # Collect all landmarks as points
            points = []
            for lm in hand_landmarks.landmark:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                points.append([x, y])

            points = np.array(points, dtype=np.int32)

            if points.shape[0] >= 5:
                cv2.fillPoly(hand_mask, [points], 255)  # Fill hand region


            # Get thumb and index finger tips for movement tracking
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Convert normalized landmarks to pixel coordinates
            thumb_tip_x, thumb_tip_y = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])
            index_tip_x, index_tip_y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])
            
            # Draw points on the frame
            cv2.circle(frame, (thumb_tip_x, thumb_tip_y), 10, (255, 0, 0), -1)
            cv2.circle(frame, (index_tip_x, index_tip_y), 10, (0, 255, 0), -1)
            
            hand_pts = []
            for lm in hand_landmarks.landmark:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                hand_pts.append(np.array([x, y]))

            current_time = time.time()
            
            # Track thumb movement to detect switch flip
            if (projected_image_box) is not None:
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

            previous_thumb_y = thumb_tip_y
            previous_thumb_x = thumb_tip_x
            
            for landmark_pt in hand_pts:
                for sw_pt in projected_switch_centers:
                    dist = np.linalg.norm(landmark_pt - sw_pt)
                    if dist < 100:  # Distance threshold in pixels
                        cv2.putText(frame, "Hand near switch — skipping detection",
                                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        return False  # Skip color detection
            
    return True

def init_qr(frame):
    global qr_scanned, saved_rvec, saved_tvec, saved_bbox, saved_data
    camera_matrix = get_camera_matrix(frame.shape)
    dist_coeffs = np.zeros((4, 1)) 
    
    #qr_scanned = False
    
    if not qr_scanned:
        data, bbox = detect_single_qr_code(frame)
        if bbox is not None and len(bbox) == 4:
            success, rvec, tvec = cv2.solvePnP(
                object_points, bbox, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if success:
                qr_scanned = True
                saved_rvec = rvec
                saved_tvec = tvec
                saved_bbox = bbox
                saved_data = data
                #draw_qr_pose_info(frame, saved_bbox, saved_data, saved_rvec, saved_tvec, camera_matrix, dist_coeffs)
    else:
        # Use the saved rvec, tvec, bbox
        draw_qr_pose_info(frame, saved_bbox, saved_data, saved_rvec, saved_tvec, camera_matrix, dist_coeffs)

def color_detect_yellow(frame):
    if projected_image_box is not None:
        for i, box in enumerate(projected_image_box):
            # Convert to np.array for math
            #print(box)
            box = np.array(box, dtype=np.int32)
            
            # Bounding rectangle
            roi_top_left = np.min(box, axis=0)
            roi_bottom_right = np.max(box, axis=0)

            # Crop ROI
            roi = frame[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            yellow_mask = cv2.inRange(hsv, lower_color, upper_color)
            
            # Apply global hand mask (remove hand regions)
            hand_mask_roi = hand_mask[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]]
            yellow_mask = cv2.bitwise_and(yellow_mask, cv2.bitwise_not(hand_mask_roi))


            # Offset box coordinates into ROI-local space
            offset_box = box - roi_top_left

            if i >= 3:
                top_mid = ((offset_box[0] + offset_box[1]) // 2)
                bottom_mid = ((offset_box[3] + offset_box[2]) // 2)

                left_poly = np.array([offset_box[0], offset_box[3], top_mid, bottom_mid], dtype=np.int32)
                right_poly = np.array([offset_box[1], offset_box[2], bottom_mid, top_mid], dtype=np.int32)

                divider_pt1 = top_mid + roi_top_left
                divider_pt2 = bottom_mid + roi_top_left
            else:
                left_mid = ((offset_box[0] + offset_box[3]) // 2)
                right_mid = ((offset_box[1] + offset_box[2]) // 2)

                top_poly = np.array([offset_box[0], offset_box[1], left_mid, right_mid], dtype=np.int32)
                bottom_poly = np.array([offset_box[2], offset_box[3], right_mid, left_mid], dtype=np.int32)

                divider_pt1 = left_mid + roi_top_left
                divider_pt2 = right_mid + roi_top_left

            # Create masks
            mask_shape = yellow_mask.shape
            mask_a = np.zeros(mask_shape, dtype=np.uint8)
            mask_b = np.zeros(mask_shape, dtype=np.uint8)

            if i >= 3:
                cv2.fillPoly(mask_a, [left_poly], 255)
                cv2.fillPoly(mask_b, [right_poly], 255)
            else:
                cv2.fillPoly(mask_a, [top_poly], 255)
                cv2.fillPoly(mask_b, [bottom_poly], 255)

            count_a = cv2.countNonZero(cv2.bitwise_and(yellow_mask, yellow_mask, mask=mask_a))
            count_b = cv2.countNonZero(cv2.bitwise_and(yellow_mask, yellow_mask, mask=mask_b))

            # Determine state
            state = "ON" if count_a > count_b else "OFF"
            
            # Confirm flip if one is pending
            if pending_flips[i]:
                if state == predicted_states[i]:
                    switch_states[i] = state  # Confirm success
                    pending_flips[i] = False
                    print(f"Switch {i+1} confirmed: {state}")
                else:
                    switch_states[i] = state
                    pending_flips[i] = False
                    print(f"Switch {i+1} denied")

            # Draw switch box
            cv2.polylines(frame, [box], isClosed=True, color=(255, 0, 0), thickness=2)

            # Draw divider line
            cv2.line(frame,
                    tuple(divider_pt1.astype(int)),
                    tuple(divider_pt2.astype(int)),
                    (0, 255, 255), 2)

            # Label
            label_pos = tuple(box[0])
            cv2.putText(frame,
                        f"Switch {i+1}: {state}",
                        (label_pos[0], label_pos[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0) if state == "ON" else (0, 0, 255),
                        2)


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    init_qr(frame)
    
    if qr_scanned and saved_rvec is not None and saved_tvec is not None:
        # Apply pose transformation to each local switch center
        switch_centers_np = np.array(switch_centers, dtype=np.float32)  # shape (N, 3)
        transformed_switches = transform_points(switch_centers_np, saved_rvec, saved_tvec)

        # Project transformed 3D points to 2D
        image_points, _ = cv2.projectPoints(transformed_switches.reshape(-1, 1, 3), np.zeros((3, 1)), np.zeros((3, 1)),
                                            get_camera_matrix(frame.shape), np.zeros((4, 1)))
        
        projected_switch_centers = [pt.ravel().astype(int) for pt in image_points]


        for i, point in enumerate(image_points):
            x, y = point.ravel().astype(int)
            cv2.circle(frame, (x, y), 6, (0, 255, 255), -1)
            cv2.putText(frame, f"SW{i+1}", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    
    track_color = hand_track_region(frame)
    
    if track_color:
        color_detect_yellow(frame)

    cv2.imshow("QR Pose Estimation with LED Regions", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('r'):
        qr_scanned = False

cap.release()
cv2.destroyAllWindows()