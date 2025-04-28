import cv2
import mediapipe as mp
import numpy as np
import qr
from zones import QR_ZONE_DEFINITIONS
from constants import COLORS
from constants import ENCODING

# Define color range for light detection (e.g., red light)
sw_color = "switch_yellow"
sw_lower_color = np.array(COLORS[sw_color][0])
sw_upper_color = np.array(COLORS[sw_color][1])

btn_color = "led_green"
btn_lower_color = np.array(COLORS[btn_color][0])
btn_upper_color = np.array(COLORS[btn_color][1])

knob_red = "knob_red_low"
knob_red_l = np.array(COLORS[knob_red][0])
knob_red_u = np.array(COLORS[knob_red][1])

knob_red_2 = "knob_red_high"
knob_red_l2 = np.array(COLORS[knob_red_2][0])
knob_red_u2 = np.array(COLORS[knob_red_2][1])

knob_green = "knob_green"
knob_green_l = np.array(COLORS[knob_green][0])
knob_green_u = np.array(COLORS[knob_green][1])

knob_blue = "knob_blue"
knob_blue_l = np.array(COLORS[knob_blue][0])
knob_blue_u = np.array(COLORS[knob_blue][1])

knob_magenta = "knob_magenta"
knob_magenta_l = np.array(COLORS[knob_magenta][0])
knob_magenta_u = np.array(COLORS[knob_magenta][1])

knob_yellow = "knob_yellow"
knob_yellow_l = np.array(COLORS[knob_yellow][0])
knob_yellow_u = np.array(COLORS[knob_yellow][1])

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
                
            hand_pts_np = np.array(hand_pts)  # shape: (N, 2)
            for module_data in qr.qr_cache.values():
                centers = np.array(module_data['centers'])  # shape: (M, 2)

                # Compute pairwise distances between all hand points and all projected centers
                dists = np.linalg.norm(hand_pts_np[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
                min_dists = np.min(dists, axis=1)

                if np.any(min_dists < 150):  # threshold
                    blocked.append(module_data['zone_def']['type'])

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
            h, w = frame.shape[:2]
            x1 = max(0, roi_top_left[0])
            y1 = max(0, roi_top_left[1])
            x2 = min(w, roi_bottom_right[0])
            y2 = min(h, roi_bottom_right[1])

            if x2 <= x1 or y2 <= y1:
                continue

            roi = frame[y1:y2, x1:x2]
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
            
            if state == "ON":
                qr.qr_cache[key]['state'][i] = 1
            else:
                qr.qr_cache[key]['state'][i] = 0
            
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
            h, w = frame.shape[:2]
            x1 = max(0, roi_top_left[0])
            y1 = max(0, roi_top_left[1])
            x2 = min(w, roi_bottom_right[0])
            y2 = min(h, roi_bottom_right[1])

            if x2 <= x1 or y2 <= y1:
                continue

            roi = frame[y1:y2, x1:x2]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # Create green mask
            mask = cv2.inRange(hsv, btn_lower_color, btn_upper_color)
            
            # Count # of green pixels
            nonzero = cv2.countNonZero(mask)
            
            label_pos = tuple(box[0])
            
            # Threshold to determine "LED is on"
            if nonzero > 500:
                qr.qr_cache[key]['state'][i] = 1
                cv2.putText(frame, f"BTN {i+1}: ON",
                        (label_pos[0], label_pos[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 0),
                        2)
            else:
                qr.qr_cache[key]['state'][i] = 0
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
        
        temp_states = [0] * len(boxes)
        for i, box in enumerate(boxes):
            box = np.array(box, dtype=np.int32)
            
            # Bounding rectangle
            roi_top_left = np.min(box, axis=0)
            roi_bottom_right = np.max(box, axis=0)
            
            # Make a mask over the ROI
            h, w = frame.shape[:2]
            x1 = max(0, roi_top_left[0])
            y1 = max(0, roi_top_left[1])
            x2 = min(w, roi_bottom_right[0])
            y2 = min(h, roi_bottom_right[1])

            if x2 <= x1 or y2 <= y1:
                continue

            roi = frame[y1:y2, x1:x2]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # Create red mask and count # of red pixels
            red_mask = cv2.inRange(hsv, knob_red_l, knob_red_u)
            red_count = cv2.countNonZero(red_mask)
            
            # Create red mask and count # of red pixels
            red_mask2 = cv2.inRange(hsv, knob_red_l2, knob_red_u2)
            red_count += cv2.countNonZero(red_mask2)
            
            # Create green mask and count # of green pixels 
            green_mask = cv2.inRange(hsv, knob_green_l, knob_green_u)
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
            
            OFF_THRESHOLD = 100
            
            counts = [OFF_THRESHOLD, red_count, green_count, blue_count, magenta_count, yellow_count]
            
            dominant_color = OFF_THRESHOLD
            dominant_color_idx = 0
            for j in range(len(counts)):
                if counts[j] > dominant_color:
                    dominant_color = counts[j]
                    dominant_color_idx = j

            state = dominant_color_idx
            
            temp_states[i] = state
            
            if state == 0:
                col = (211, 211, 211)
            elif state == 1:
                col = (0, 0, 255)
            elif state == 2:
                col = (0, 255, 0)
            elif state == 3:
                col = (255, 255, 0)
            elif state == 4:
                col = (255, 0, 255)
            else:
                col = (5, 226, 252)

            # Draw rectangle and status on original frame
            cv2.polylines(frame, [box], isClosed=True, color=(255, 0, 0), thickness=2)
            
            text = f"KNOB {i+1}: {state}"
            
            label_pos = tuple(box[0])
            position = (label_pos[0], label_pos[1] - 10)
            
            cv2.putText(frame, text,
            (position[0] + 1, position[1] + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA)
            
            cv2.putText(frame, text,
                        (label_pos[0], label_pos[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        col,
                        2)
        
        encoded = base6_to_decimal(temp_states)

        qr.qr_cache[key]['state'] = encoded

def base6_to_decimal(list):
    if ((len(list) % 2) != 0) or (len(list) < 2):
        print("Incorrect list size")
        return -1
    
    decimal_list = []

    for i in range(1, len(list), 2):
        MSD = list[i]
        LSD = list[i - 1]
        decimal_list.append(ENCODING.get((MSD, LSD)))
    
    if -1 in decimal_list:
        print("State error")
        return -1
    
    return decimal_list