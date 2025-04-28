import cv2
import numpy as np
import detect
from constants import FOCAL_LENGTH, OBJECT_POINTS
from zones import QR_ZONE_DEFINITIONS

qr_cache = {}
distance = False

def get_camera_matrix(frame_shape):
    cx = frame_shape[1] / 2
    cy = frame_shape[0] / 2
    return np.array([
        [FOCAL_LENGTH, 0, cx],
        [0, FOCAL_LENGTH, cy],
        [0, 0, 1]
    ], dtype=np.float64)

def detect_multiple_qr_codes(frame):
    qr_detector = cv2.QRCodeDetector()
    retval, decoded_info, points, _ = qr_detector.detectAndDecodeMulti(frame)

    qr_list = []
    if retval and points is not None:
        for data, corner_pts in zip(decoded_info, points):
            if data:
                qr_list.append((data, np.array(corner_pts, dtype=np.float32)))
    return qr_list

def transform_points(points, rvec, tvec):
    # Convert rvec to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    transformed = []
    for pt in points:
        world_pt = R @ pt.reshape(3, 1) + tvec
        transformed.append(world_pt.flatten())
    return np.array(transformed, dtype=np.float32)

def transform_and_project_zone(zone_definition, rvec, tvec, camera_matrix, dist_coeffs):
    local_centers = np.array(zone_definition["centers"], dtype=np.float32)
    w, h = zone_definition["size"]
    half_w, half_h = w / 2, h / 2

    # Create local corner box
    box_local = np.array([
        [-half_w, -half_h, 0],
        [ half_w, -half_h, 0],
        [ half_w,  half_h, 0],
        [-half_w,  half_h, 0]
    ], dtype=np.float32)

    all_projected_boxes = []

    for center in local_centers:
        # Move box to local center
        box = box_local + center

        # Transform to camera coordinates
        R, _ = cv2.Rodrigues(rvec)
        world_points = (R @ box.T + tvec).T

        # Project to image
        img_points, _ = cv2.projectPoints(world_points, np.zeros((3,1)), np.zeros((3,1)), camera_matrix, dist_coeffs)
        projected_box = img_points.reshape(-1, 2).astype(int)
        
        all_projected_boxes.append(projected_box)

    return all_projected_boxes

def draw_qr_and_zones(frame, boxes, qr_data, bbox):
    # Draw bounding box
    for i in range(len(bbox)):
        pt1 = tuple(bbox[i].astype(int))
        pt2 = tuple(bbox[(i + 1) % len(bbox)].astype(int))
        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    # Label QR code
    center = np.mean(bbox, axis=0).astype(int)
    cv2.putText(frame, f"QR: {qr_data}", (center[0], center[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    for i, zone in enumerate(boxes):
        cv2.polylines(frame, [zone], isClosed=True, color=(255, 255, 0), thickness=2)

def draw_projected_zones(frame):
    camera_matrix = get_camera_matrix(frame.shape)
    
    for code in list(qr_cache.keys()):
        zone_def = qr_cache[code].get('zone_def')
        rvec = qr_cache[code].get('rvec')
        tvec = qr_cache[code].get('tvec')
        
        # Apply pose transformation to each local center
        switch_centers_np = np.array(zone_def["centers"], dtype=np.float32)
        transformed_switches = transform_points(switch_centers_np, rvec, tvec)
        
        # Project transformed 3D points to 2D
        image_points, _ = cv2.projectPoints(transformed_switches.reshape(-1, 1, 3), np.zeros((3, 1)), np.zeros((3, 1)), camera_matrix, np.zeros((4, 1)))
        
        projected_centers = [pt.ravel().astype(int) for pt in image_points]
        
        if code == "1":
            for i, point in enumerate(image_points):
                x, y = point.ravel().astype(int)
                cv2.circle(frame, (x, y), 6, (0, 255, 255), -1)
        elif code == "2":
            for i, point in enumerate(image_points):
                x, y = point.ravel().astype(int)
                cv2.circle(frame, (x, y), 6, (0, 255, 255), -1)
                cv2.putText(frame, f"BTN{i+1}", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        elif code == "3":
            for i, point in enumerate(image_points):
                x, y = point.ravel().astype(int)
                cv2.circle(frame, (x, y), 6, (0, 255, 255), -1)
        
        qr_cache[code]['centers'] = projected_centers

def init_qr(frame):
    global qr_cache
    camera_matrix = get_camera_matrix(frame.shape)
    dist_coeffs = np.zeros((4, 1))
    
    qr_list = detect_multiple_qr_codes(frame)
    
    if (len(qr_cache) < len(qr_list)) or (len(qr_cache) == 0):
        # Clear all saved boxes
        qr_cache = {}
        
        for qr_data, bbox in qr_list:
            if len(bbox) == 4:
                success, rvec, tvec = cv2.solvePnP(
                    OBJECT_POINTS, bbox, camera_matrix, dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                if success:
                    zone_def = QR_ZONE_DEFINITIONS[qr_data]
                    boxes = transform_and_project_zone(zone_def, rvec, tvec, camera_matrix, dist_coeffs)
                    
                    if boxes == []:
                        continue
                    
                    qr_cache[qr_data] = {
                    'bbox': bbox, # Bounding box of the QR code
                    'rvec': rvec,
                    'tvec': tvec,
                    'boxes': boxes, # Projected boxes
                    'zone_def': zone_def,
                    'state' : [0] * len(zone_def.get("centers")),
                    'centers' : [0] * len(zone_def.get("centers"))
                    }
    else:
        # Use the saved rvec, tvec, bbox
        keylist = list(qr_cache.keys())
        for key in keylist:
            zone_def = qr_cache[key].get('zone_def')
            boxes = qr_cache[key].get('boxes')
            bbox = qr_cache[key].get('bbox')
            
            draw_qr_and_zones(frame, boxes, key, bbox)