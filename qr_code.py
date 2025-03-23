import cv2
import numpy as np
from collections import deque

QR_SIZE_CM = 3
FOCAL_LENGTH = 442.98

N = 10
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

def draw_qr_pose_info(frame, bbox, data, rvec, tvec):
    for i in range(len(bbox)):
        pt1 = tuple(bbox[i].astype(int))
        pt2 = tuple(bbox[(i + 1) % len(bbox)].astype(int))
        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    center = np.mean(bbox, axis=0).astype(int)

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

    yaw, pitch, roll = angles[1], angles[0], angles[2]
    z = tvec[2][0]  # Depth in cm

    depth_history.append(z)
    yaw_history.append(yaw)
    pitch_history.append(pitch)
    roll_history.append(roll)

    avg_depth = sum(depth_history) / len(depth_history)
    avg_yaw = sum(yaw_history) / len(yaw_history)
    avg_pitch = sum(pitch_history) / len(pitch_history)
    avg_roll = sum(roll_history) / len(roll_history)

    cv2.circle(frame, tuple(center), 5, (0, 0, 255), -1)

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

    print(f"QR Code: {data}")
    print(f"Position: x={tvec[0][0]:.2f} cm, y={tvec[1][0]:.2f} cm, z={avg_depth:.2f} cm")
    print(f"Orientation: Yaw={avg_yaw:.2f}°, Pitch={avg_pitch:.2f}°, Roll={avg_roll:.2f}°")
    for i, point in enumerate(bbox):
        print(f"Corner {i + 1}: (X: {point[0]:.1f}, Y: {point[1]:.1f})")
    print("-" * 40)

cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    camera_matrix = get_camera_matrix(frame.shape)
    dist_coeffs = np.zeros((4, 1))  # Assuming no distortion

    data, bbox = detect_single_qr_code(frame)
    if bbox is not None and len(bbox) == 4:
        success, rvec, tvec = cv2.solvePnP(
            object_points, bbox, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if success:
            draw_qr_pose_info(frame, bbox, data, rvec, tvec)

    cv2.imshow("QR Pose Estimation (Smoothed)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
