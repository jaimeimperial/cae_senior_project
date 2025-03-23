import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

colors = {
    "green" : ([50, 0, 55], [86, 255, 255]),
    "white" : ([0,0,240], [255, 15, 255])
}

# Define color range for light detection (e.g., red light)
color = "white"
lower_color = np.array(colors[color][0])    # Lower HSV range for green
upper_color = np.array(colors[color][1])      # Upper HSV range for green

# Initialize video capture
cap = cv2.VideoCapture(0)

# To store previous hand position for movement analysis
previous_thumb_y = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally for natural interaction
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Process frame with Mediapipe
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
                        print("Switch flip down detected")
                    else:
                        print("Switch flip up detected")

            previous_thumb_y = thumb_tip_y

    mask = cv2.inRange(frame_hsv, lower_color, upper_color)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Track if the light is on or off
    light_on = False
    for contour in contours:
        area = cv2.contourArea(contour)
        if 1000 > area > 500:  # Adjust area threshold to filter noise
            light_on = True
            # Draw a rectangle around the detected light
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Light ON", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if not light_on:
        cv2.putText(frame, "Light OFF", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Gesture and Light Tracking", frame)

    # Break loop on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
