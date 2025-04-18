import cv2
import mediapipe as mp
import numpy as np
from scipy import stats
from collections import deque
import time
import qr
import zones
import detect

# Main video loop
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    qr.init_qr(frame)
    qr.draw_projected_zones(frame)

    # Detect I/O states if not occluded
    blocked = detect.hand_track_region(frame)
    detection_map = {
        "Switch": detect.detect_switch,
        "Button": detect.detect_button,
        #"Knob": detect.detect_knob
    }
    for name, func in detection_map.items():
        if name not in blocked:
            func(frame)

    
    cv2.imshow("Multi-QR Zone Detection", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        qr.qr_cache = {}

cap.release()
cv2.destroyAllWindows()
