import cv2
import mediapipe as mp
import numpy as np
from collections import deque

QR_SIZE_CM = 3               # Size of QR code (cm)
FOCAL_LENGTH = 1238.40

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
previous_thumb_y = None

colors = {
    "green" : ([50, 0, 55], [86, 255, 255]),
    "white" : ([0,0,240], [255, 15, 255])
}

# Switch Regions
z_offset = 0.5  # Dynamic Z-distance in front of QR

switch_centers = [
        [1.0, 6.0, z_offset],
        [3.3, 6.0, z_offset],
        [6.3, 6.0, z_offset],
        [1.8, 9.5, z_offset],
        [6.0, 9.5, z_offset],
    ]
switch_size = (2.0, 2.0)  # Width and height

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
    for i, box in enumerate(led_boxes_local):
        image_box, _ = cv2.projectPoints(box, rvec, tvec, camera_matrix, dist_coeffs)
        image_box = image_box.reshape(-1, 2).astype(int)

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

    camera_matrix = get_camera_matrix(frame.shape)
    dist_coeffs = np.zeros((4, 1)) 

    results = hands.process(frame_rgb)

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
            if previous_thumb_y is not None:
                movement = thumb_tip_y - previous_thumb_y
                if abs(movement) > 20:  # Threshold for detecting significant movement
                    if movement > 0:
                        for box in led_boxes_local:
                            if (box[0][1] < thumb_tip_y < box[3][1]): # and (box[0][0]< thumb_tip_x < box[1][0]):
                                print("Switch flip down detected")
                    else:
                        for box in led_boxes_local:
                            if (box[0][1] < thumb_tip_y < box[3][1]): # and (box[0][0]< thumb_tip_x < box[1][0]):
                                print("Switch flip up detected")
            previous_thumb_y = thumb_tip_y


            

    data, bbox = detect_single_qr_code(frame)
    if bbox is not None and len(bbox) == 4:
        success, rvec, tvec = cv2.solvePnP(
            object_points, bbox, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if success:
            draw_qr_pose_info(frame, bbox, data, rvec, tvec, camera_matrix, dist_coeffs)

    cv2.imshow("QR Pose Estimation with LED Regions", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
