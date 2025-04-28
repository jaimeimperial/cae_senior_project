# Define your zone types
QR_ZONE_DEFINITIONS = {
    "1": {  # Switches
        "type": "switch",
        "centers": [
            [1.3, 6.3, 0.5],
            [3.5, 6.3, 0.5],
            [6.5, 6.3, 0.5],
            [2.0, 9.7, 0.5],
            [6.2, 9.7, 0.5],
        ],
        "size": (2.5, 3.0)
    },
    "2": {  # Buttons
        "type": "button",
        "centers": [
            [4.4, 1.0, 0.5],
            [5.9, 1.0, 0.5],
            [7.5, 1.0, 0.5],
            [4.7, 2.8, 0.5],
            [5.9, 2.5, 0.5],
            [7.6, 2.3, 0.5],
        ],
        "size": (1.0, 1.0)
    },
    "3": {  # Knob
        "type": "knob",
        "centers": [
            [5.1, 2.1, 0.5], # Bottom left
            [5.05, 0.8, 0.5], # Top Left
            [7.2, 2.2, 0.5], # Bottom right
            [7.3, 0.9, 0.5], # Top Right
        ],
        "size": (1.0, 1.0),
    }
}