import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Show the HSV values when clicking on a pixel
    def pick_color(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pixel = hsv_frame[y, x]
            print(f"HSV Value at ({x}, {y}): {pixel}")

    cv2.imshow("HSV Picker", hsv_frame)
    cv2.setMouseCallback("HSV Picker", pick_color)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
