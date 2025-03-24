import cv2
import mediapipe as mp
import numpy as np
from collections import deque

QR_SIZE_CM = 3               # Size of QR code (cm)
FOCAL_LENGTH = 1238.40

colors = {
    'test_purple' :    [[110, 50, 130],    [130, 80, 190]],
    'test_yellow' :    [[200, 115, 180],    [235, 130, 205]],
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

# Variables to hold points
old_points = None
old_gray = None
roi_box = (600, 400, 100, 100)  # x, y, w, h of switch region

# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

global qr_scanned, saved_rvec, saved_tvec, saved_bbox, saved_data
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

# Switch Regions
z_offset = 0.5  # Dynamic Z-distance in front of QR

switch_centers = [
        [1.0, 6.0, z_offset],
        [3.3, 6.0, z_offset],
        [6.3, 6.0, z_offset],
        [1.8, 9.5, z_offset],
        [6.0, 9.5, z_offset],
    ]
switch_size = (2.5, 3.0)  # Width and height

led_boxes_local = []
for cx, cy, cz in switch_centers:
    w, h = switch_size
    box = np.array([
        [cx - w/2, cy - h/2, cz],
        [cx + w/2, cy - h/2, cz],
        [cx + w/2, cy + h/2, cz],
        [cx - w/2, cy + h/2, cz]
    ], dtype=np.float32)
    led_boxes_local.append(box)

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

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally for natural interaction
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(frame_hsv, lower_color, upper_color)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    camera_matrix = get_camera_matrix(frame.shape)
    dist_coeffs = np.zeros((4, 1)) 

    results = hands.process(frame_rgb)

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
    else:
        # Use the saved rvec, tvec, bbox
        draw_qr_pose_info(frame, saved_bbox, saved_data, saved_rvec, saved_tvec, camera_matrix, dist_coeffs)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get thumb and index finger tips for movement tracking
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Convert normalized landmarks to pixel coordinates
            thumb_tip_x, thumb_tip_y = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])
            index_tip_x, index_tip_y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])

            # Draw points on the frame
            cv2.circle(frame, (thumb_tip_x, thumb_tip_y), 10, (255, 0, 0), -1)
            cv2.circle(frame, (index_tip_x, index_tip_y), 10, (0, 255, 0), -1)

            # Track thumb movement to detect switch flip
            if (projected_image_box) is not None:
                for i, box in enumerate(projected_image_box):
                    if (previous_thumb_y is not None) and (previous_thumb_x is not None):
                        if (i <= 2):
                            t_movement = thumb_tip_y - previous_thumb_y
                            if abs(t_movement) > 10:  # Threshold for detecting significant movement
                                if t_movement > 0:
                                    if in_box(box, thumb_tip_x, thumb_tip_y) and in_box(box, index_tip_x, index_tip_y):
                                        print("Switch", i+1, "down detected")
                                else:
                                    if in_box(box, thumb_tip_x, thumb_tip_y) and in_box(box, index_tip_x, index_tip_y):
                                        print("Switch", i+1, "up detected")
                        if (i > 2):
                            t_movement = thumb_tip_x - previous_thumb_x
                            if abs(t_movement) > 10:  # Threshold for detecting significant movement
                                if t_movement > 0:
                                    if in_box(box, thumb_tip_x, thumb_tip_y) and in_box(box, index_tip_x, index_tip_y):
                                        print("Switch", i+1, "right detected")
                                else:
                                    if in_box(box, thumb_tip_x, thumb_tip_y) and in_box(box, index_tip_x, index_tip_y):
                                        print("Switch", i+1, "left detected")

            previous_thumb_y = thumb_tip_y
            previous_thumb_x = thumb_tip_x
            
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Draw the ROI box
    x, y, w, h = roi_box
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    if old_points is None or len(old_points) < 5:
        roi = frame_gray[y:y + h, x:x + w]
        old_points = cv2.goodFeaturesToTrack(roi, mask=None, **feature_params)
        if old_points is not None:
            # Offset keypoints to absolute frame coords
            old_points += np.array([[x, y]], dtype=np.float32)
        old_gray = frame_gray.copy()
    else:
        # Calculate Optical Flow
        new_points, status, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_points, None, **lk_params)

        if new_points is not None:
            good_old = old_points[status == 1]
            good_new = new_points[status == 1]

            # Draw points and compute motion
            motion_vectors = []
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                motion_vectors.append((a - c, b - d))
                cv2.circle(frame, (int(a), int(b)), 3, (0, 255, 0), -1)
                cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 255), 1)

            # Average motion
            if motion_vectors:
                avg_dx = sum([dx for dx, dy in motion_vectors]) / len(motion_vectors)
                avg_dy = sum([dy for dx, dy in motion_vectors]) / len(motion_vectors)

                if abs(avg_dy) > abs(avg_dx):
                    if avg_dy > 2:
                        cv2.putText(frame, "Switch moved DOWN", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        print("Switch Down")
                    elif avg_dy < -2:
                        cv2.putText(frame, "Switch moved UP", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        print("Switch up")
                else:
                    if avg_dx > 2:
                        cv2.putText(frame, "Switch moved RIGHT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    elif avg_dx < -2:
                        cv2.putText(frame, "Switch moved LEFT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

            # Update for next frame
            old_gray = frame_gray.copy()
            old_points = good_new.reshape(-1, 1, 2)

    cv2.imshow("QR Pose Estimation with LED Regions", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        qr_scanned = False

cap.release()
cv2.destroyAllWindows()
