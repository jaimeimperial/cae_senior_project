import cv2
import numpy as np

colors = {
    'test_purple' :    [[110, 50, 130],     [130, 80, 190]],
    'test_yellow' :    [[20, 60, 60],       [30, 255, 255]],
    'test_green' :     [[30, 0, 240],       [100, 15, 255]],
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
color = "test_green"
lower_color = np.array(colors[color][0])
upper_color = np.array(colors[color][1])

# Define regions of interest (top-left and bottom-right corners)
regions = {
    "LED 1": ((700, 200), (750, 250)),
    "LED 2": ((800, 200), (850, 250)),
}

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create red mask (combine two ranges)
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Check each region
    for name, ((x1, y1), (x2, y2)) in regions.items():
        region_mask = mask[y1:y2, x1:x2]
        nonzero = cv2.countNonZero(region_mask)
        
        # Threshold to determine "LED is on"
        if nonzero > 50:
            status = "ON"
            color = (0, 255, 0)
        else:
            status = "OFF"
            color = (0, 0, 255)

        # Draw rectangle and status on original frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{name}: {status}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show live feed
    cv2.imshow("LED Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
