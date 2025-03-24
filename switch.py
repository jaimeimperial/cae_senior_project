import cv2
import numpy as np

# ROI for your switch (adjust for your image)
roi_top_left = (300, 300)
roi_bottom_right = (450, 450)

def detect_yellow_toggle(frame):
    roi = frame[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Define yellow color range in HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Divide mask into top and bottom halves
    height = mask.shape[0]
    top_half = mask[:height//2, :]
    bottom_half = mask[height//2:, :]

    top_yellow = cv2.countNonZero(top_half)
    bottom_yellow = cv2.countNonZero(bottom_half)

    # Determine state
    if top_yellow > bottom_yellow:
        return "ON"
    else:
        return "OFF"

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    state = detect_yellow_toggle(frame)

    # Visuals
    cv2.putText(frame, f"Switch State: {state}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.rectangle(frame, roi_top_left, roi_bottom_right, (255, 0, 0), 2)

    cv2.imshow("Yellow Toggle Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
