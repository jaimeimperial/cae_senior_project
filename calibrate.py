import cv2
import numpy as np

KNOWN_DISTANCE_CM = 16.6     # Distance from camera to QR code (cm)
KNOWN_WIDTH_CM = 3      # Real-world width of QR code (cm)

def detect_qr_code_and_measure_width(frame):
    qr_decoder = cv2.QRCodeDetector()
    data, bbox, _ = qr_decoder.detectAndDecode(frame)

    if bbox is not None and data:
        bbox = bbox.astype(int)
        width_px = np.linalg.norm(bbox[0][0] - bbox[0][1])
        return data, width_px, bbox[0]
    return None, None, None

def draw_qr_box(frame, corners, width_px, focal_length):
    for i in range(len(corners)):
        pt1 = tuple(corners[i])
        pt2 = tuple(corners[(i+1) % len(corners)])
        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    text = f"Width: {width_px:.2f} px | Focal Len: {focal_length:.2f}"
    cv2.putText(frame, text, (corners[0][0], corners[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
print("Hold your printed QR code at a known distance (e.g. 30 cm) from the camera.")
print("Press 's' to capture and calculate focal length. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    data, width_px, corners = detect_qr_code_and_measure_width(frame)

    if width_px:
        # Compute focal length: F = (P * D) / W
        focal_length = (width_px * KNOWN_DISTANCE_CM) / KNOWN_WIDTH_CM
        draw_qr_box(frame, corners, width_px, focal_length)

    cv2.imshow("Focal Length Calibration", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and width_px:
        print(f"[INFO] Captured width: {width_px:.2f} px")
        print(f"[INFO] Focal Length = {focal_length:.2f} px")
        break
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
