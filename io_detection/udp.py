import socket
import numpy as np

sock_io = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

serverAddressPort_io = ("127.0.0.1", 5054)

def sorted_qr(qr_cache):
    module_positions = []

    for qr_id, qr_data in qr_cache.items():
        bbox = qr_data.get("bbox")

        # Average X of bbox points
        x_coords = [pt[0] for pt in bbox if isinstance(pt, (list, tuple)) and pt != 0]
        
        try:
            # Extract average X value
            x_coords = [pt[0] for pt in bbox if isinstance(pt, (list, np.ndarray))]
            avg_x = sum(x_coords) / len(x_coords)
            module_positions.append((qr_id, avg_x))
        except Exception as e:
            print(f"Error processing bbox for {qr_id}: {e}")
            continue

    # Sort by average X coordinate
    sorted_modules = sorted(module_positions, key=lambda x: x[1])
    sorted_ids = [qr_id for qr_id, _ in sorted_modules]
    

    return sorted_ids
    

def send_info(qr_cache):
    io_state = {}
    ordered_keys = sorted_qr(qr_cache)[::-1]
    
    for key in ordered_keys:
        state = qr_cache[key].get('state')
        
        if state is None or -1 in state:
            print("Bad Value")
        else:
            io_state[key] = state

            
    print(io_state)

    sock_io.sendto(str.encode(str(io_state)), serverAddressPort_io)