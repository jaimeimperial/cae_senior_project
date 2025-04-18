import cv2
import mediapipe as mp
import numpy as np
from scipy import stats
from collections import deque
import time
import qr
from zones import QR_ZONE_DEFINITIONS
from constants import COLORS
from constants import ENCODING

projected_switch_centers = []
projected_button_centers = []

# Define color range for light detection (e.g., red light)
sw_color = "switch_yellow"
sw_lower_color = np.array(COLORS[sw_color][0])
sw_upper_color = np.array(COLORS[sw_color][1])

btn_color = "led_green"
btn_lower_color = np.array(COLORS[btn_color][0])
btn_upper_color = np.array(COLORS[btn_color][1])

knob_red = "knob_red"
knob_red_l = np.array(COLORS[knob_red][0])
knob_red_u = np.array(COLORS[knob_red][1])

knob_blue = "knob_blue"
knob_blue_l = np.array(COLORS[knob_blue][0])
knob_blue_u = np.array(COLORS[knob_blue][1])

knob_magenta = "knob_magenta"
knob_magenta_l = np.array(COLORS[knob_magenta][0])
knob_magenta_u = np.array(COLORS[knob_magenta][1])

knob_yellow = "knob_yellow"
knob_yellow_l = np.array(COLORS[knob_yellow][0])
knob_yellow_u = np.array(COLORS[knob_yellow][1])

hand_mask = np.zeros((1080, 1920), dtype=np.uint8)  # Update if frame size changes

zone_def = QR_ZONE_DEFINITIONS.get("1")
switch_centers = zone_def["centers"]

previous_thumb_y = None
previous_thumb_x = None
previous_y = None
previous_x = None


last_flip_time = [0] * len(switch_centers)  # One timestamp per switch
debounce_delay = 0.8  # seconds
switch_states = ["OFF"] * len(switch_centers)        # Confirmed states
predicted_states = ["OFF"] * len(switch_centers)     # Prediction from hand tracking
pending_flips = [False] * len(switch_centers)        # Waiting for confirmation

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

def in_box(box, x, y):
    right_edge = max(box[1][0], box[2][0])
    top_edge = max(box[2][1], box[3][1])
    left_edge = min(box[0][0], box[3][0])
    bottom_edge = min(box[0][1], box[1][1])
    if (bottom_edge <= y <= top_edge) and (left_edge <= x <= right_edge):
        return True
    else:
        return False

def hand_track_region(frame):
    global previous_thumb_x, previous_thumb_y, distance
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(frame_rgb)
    
    blocked = []
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Collect all landmarks as points
            points = []
            for lm in hand_landmarks.landmark:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                points.append([x, y])

            points = np.array(points, dtype=np.int32)

            # Get thumb and index finger tips for movement tracking
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Convert normalized landmarks to pixel coordinates
            thumb_tip_x, thumb_tip_y = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])
            index_tip_x, index_tip_y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])
            
            # Draw points on the frame
            cv2.circle(frame, (thumb_tip_x, thumb_tip_y), 10, (255, 0, 0), -1)
            cv2.circle(frame, (index_tip_x, index_tip_y), 10, (0, 255, 0), -1)
            
            # Find if switch/buttons are occluded
            hand_pts = []
            for lm in hand_landmarks.landmark:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                hand_pts.append(np.array([x, y]))

            for landmark_pt in hand_pts:
                for sw_pt, bu_pt in zip(projected_switch_centers, projected_button_centers):
                    sw_dist = np.linalg.norm(landmark_pt - sw_pt)
                    bu_dist = np.linalg.norm(landmark_pt - bu_pt)
                    if (sw_dist < 100):  # Distance threshold in pixels
                        blocked.append("Switch")
                    if (bu_dist < 100):  # Distance threshold in pixels
                        blocked.append("Button")
                    
            if blocked != []:
                cv2.putText(frame, "Occlusion Detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return blocked

def detect_switch(frame):
    for key in list(qr.qr_cache.keys()):
        # Check if the current QR code is a switch
        zone_def = qr.qr_cache[key].get('zone_def')
        
        if zone_def.get('type') != 'switch':
            continue
        
        # Iterate through the projected boxes per key
        boxes = qr.qr_cache[key].get('boxes')
        for i, box in enumerate(boxes):
            # Convert to np.array for math
            box = np.array(box, dtype=np.int32)
            
            # Bounding rectangle
            roi_top_left = np.min(box, axis=0)
            roi_bottom_right = np.max(box, axis=0)
            
            # Crop ROI
            roi = frame[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            yellow_mask = cv2.inRange(hsv, sw_lower_color, sw_upper_color)
            
            # Offset box coordinates into ROI-local space
            offset_box = box - roi_top_left
            if i >= 3:
                top_mid = ((offset_box[0] + offset_box[1]) // 2)
                bottom_mid = ((offset_box[3] + offset_box[2]) // 2)
                left_poly = np.array([offset_box[0], offset_box[3], top_mid, bottom_mid], dtype=np.int32)
                right_poly = np.array([offset_box[1], offset_box[2], bottom_mid, top_mid], dtype=np.int32)
                divider_pt1 = top_mid + roi_top_left
                divider_pt2 = bottom_mid + roi_top_left
            else:
                left_mid = ((offset_box[0] + offset_box[3]) // 2)
                right_mid = ((offset_box[1] + offset_box[2]) // 2)
                top_poly = np.array([offset_box[0], offset_box[1], left_mid, right_mid], dtype=np.int32)
                bottom_poly = np.array([offset_box[2], offset_box[3], right_mid, left_mid], dtype=np.int32)
                divider_pt1 = left_mid + roi_top_left
                divider_pt2 = right_mid + roi_top_left
                
            # Create masks
            mask_shape = yellow_mask.shape
            mask_a = np.zeros(mask_shape, dtype=np.uint8)
            mask_b = np.zeros(mask_shape, dtype=np.uint8)
            if i >= 3:
                cv2.fillPoly(mask_a, [left_poly], 255)
                cv2.fillPoly(mask_b, [right_poly], 255)
            else:
                cv2.fillPoly(mask_a, [top_poly], 255)
                cv2.fillPoly(mask_b, [bottom_poly], 255)
            count_a = cv2.countNonZero(cv2.bitwise_and(yellow_mask, yellow_mask, mask=mask_a))
            count_b = cv2.countNonZero(cv2.bitwise_and(yellow_mask, yellow_mask, mask=mask_b))
            
            # Determine state
            state = "ON" if count_a > count_b else "OFF"
            
            # Draw switch box
            cv2.polylines(frame, [box], isClosed=True, color=(255, 0, 0), thickness=2)
            # Draw divider line
            cv2.line(frame,
                    tuple(divider_pt1.astype(int)),
                    tuple(divider_pt2.astype(int)),
                    (0, 255, 255), 2)
            # Label
            label_pos = tuple(box[0])
            cv2.putText(frame, f"Switch {i+1}: {state}", (label_pos[0], label_pos[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0) if state == "ON" else (0, 0, 255),
                        2)
            
def detect_button(frame):
    for key in list(qr.qr_cache.keys()):
        # Check if the current QR code is a button
        zone_def = qr.qr_cache[key].get('zone_def')
        if zone_def.get('type') != 'button':
            continue
        
        boxes = qr.qr_cache[key].get('boxes')
        for i, box in enumerate(boxes):
            box = np.array(box, dtype=np.int32)
            
            # Bounding rectangle
            roi_top_left = np.min(box, axis=0)
            roi_bottom_right = np.max(box, axis=0)
            
            # Make a mask over the ROI
            roi = frame[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # Create green mask
            mask = cv2.inRange(hsv, btn_lower_color, btn_upper_color)
            
            # Count # of green pixels
            nonzero = cv2.countNonZero(mask)
            print(i, nonzero)
            
            label_pos = tuple(box[0])
            # Threshold to determine "LED is on"
            if nonzero > 1000:
                cv2.putText(frame, f"BTN {i+1}: ON",
                        (label_pos[0], label_pos[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 0),
                        2)
            else:
                cv2.putText(frame, f"BTN {i+1}: OFF",
                        (label_pos[0], label_pos[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 0, 255),
                        2)

            # Draw rectangle and status on original frame
            cv2.polylines(frame, [box], isClosed=True, color=(255, 0, 0), thickness=2)
            
            
            
def detect_knob(frame):
    for key in list(qr.qr_cache.keys()):
        # Check if the current QR code is a button
        zone_def = qr.qr_cache[key].get('zone_def')
        if zone_def.get('type') != 'knob':
            continue
        
        boxes = qr.qr_cache[key].get('boxes')
        
        knobs = {
            1 : [],
            2 : []
        }
        
        for i, box in enumerate(boxes):
            box = np.array(box, dtype=np.int32)
            
            # Bounding rectangle
            roi_top_left = np.min(box, axis=0)
            roi_bottom_right = np.max(box, axis=0)
            
            # Make a mask over the ROI
            roi = frame[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # Create red mask and count # of red pixels
            red_mask = cv2.inRange(hsv, knob_red_l, knob_red_u)
            red_count = cv2.countNonZero(red_mask)
            
            # Create green mask and count # of green pixels 
            green_mask = cv2.inRange(hsv, btn_lower_color, btn_upper_color)
            green_count = cv2.countNonZero(green_mask)

            # Create blue mask and count # of blue pixels 
            blue_mask = cv2.inRange(hsv, knob_blue_l, knob_blue_u)
            blue_count = cv2.countNonZero(blue_mask)

            # Create magenta mask and count # of magenta pixels 
            magenta_mask = cv2.inRange(hsv, knob_magenta_l, knob_magenta_u)
            magenta_count = cv2.countNonZero(magenta_mask)

            # Create yellow mask and count # of yellow pixels 
            yellow_mask = cv2.inRange(hsv, knob_yellow_l, knob_yellow_u)
            yellow_count = cv2.countNonZero(yellow_mask)
            
            OFF_THRESHOLD = 300
            
            counts = [OFF_THRESHOLD, red_count, green_count, blue_count, magenta_count, yellow_count]
            
            dominant_color = OFF_THRESHOLD
            dominant_color_idx = 0
            for i in range(len(counts)):
                if counts[i] > dominant_color:
                    dominant_color = counts[i]
                    dominant_color_idx = i

            state = dominant_color_idx
            # Get the led box state and update the appropriate slot in knobs dict
            knob_num = i // 2 + 1
            slot = i % 2
            knobs[knob_num][slot] = state
            
            if state == 0:
                color = (211, 211, 211)
            elif state == 1:
                color = (255, 255, 255)
            elif state == 2:
                color = (0, 255, 0)
            elif state == 3:
                color = (0, 0, 255)
            elif state == 4:
                color = (255, 0, 255)
            else:
                color = (255, 255, 0)

            # Draw rectangle and status on original frame
            cv2.polylines(frame, [box], isClosed=True, color=(255, 0, 0), thickness=2)
            label_pos = tuple(box[0])
            
            cv2.putText(frame, f"KNOB {i+1}: {state}",
                        (label_pos[0], label_pos[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        2)
        
        knob1 = ENCODING[tuple(knobs[1])]
        knob2 = ENCODING[tuple(knobs[2])]