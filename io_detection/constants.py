import numpy as np

QR_SIZE_CM = 3               # Size of QR code (cm)
FOCAL_LENGTH = 1238.40

# QR and zone definitions
OBJECT_POINTS = np.array([
    [0, 0, 0],
    [QR_SIZE_CM, 0, 0],
    [QR_SIZE_CM, QR_SIZE_CM, 0],
    [0, QR_SIZE_CM, 0]
], dtype=np.float32)

COLORS = {
    'switch_yellow' :     [[20, 60, 60],    [30, 255, 255]],
    'led_green' :       [[0, 0, 240],       [90, 15, 255]],
    'knob_red' :        [[60, 0, 240],       [80, 15, 255]],
    'knob_blue' :       [[60, 0, 240],       [80, 15, 255]],
    'knob_magenta' :    [[60, 0, 240],       [80, 15, 255]],
    'knob_yellow' :     [[60, 0, 240],       [80, 15, 255]],
}

# OFF       = 0
# RED       = 1
# GREEN     = 2
# BLUE      = 3
# MAGENTA   = 4
# YELLOW    = 5
ENCODING = {
    (0, 0): 0,
    (0, 1): 1,
    (0, 2): 2,
    (0, 3): 3,
    (0, 4): 4,
    (0, 5): 5,
    (1, 0): 6,
    (1, 1): 7,
    (1, 2): 8,
    (1, 3): 9,
    (1, 4): 10,
    (1, 5): 11,
    (2, 0): 12,
    (2, 1): 13,
    (2, 2): 14,
    (2, 3): 15,
    (2, 4): 16,
    (2, 5): 17,
    (3, 0): 18,
    (3, 1): 19,
}