import cv2
import qr
import detect
import udp

# Main video loop
cam_id = 0
cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
cap.set(cv2.CAP_PROP_EXPOSURE, 5)
cap.set(cv2.CAP_PROP_GAIN, 0)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 5)
cap.set(cv2.CAP_PROP_FPS, 144)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    qr.init_qr(frame)
    qr.draw_projected_zones(frame)

    # Detect I/O states if not occluded
    blocked = detect.hand_track_region(frame)
    detection_map = {
        "switch": detect.detect_switch,
        "button": detect.detect_button,
        "knob"  : detect.detect_knob
    }
    for name, func in detection_map.items():
        if name not in blocked:
            func(frame)

    udp.send_info(qr.qr_cache)

    cv2.imshow("Multi-QR Zone Detection", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        qr.qr_cache = {}

cap.release()
cv2.destroyAllWindows()
