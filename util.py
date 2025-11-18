import numpy as np


# The angle is created between 8 and 6 and 5 of the index finger
def get_angle(a, b, c):
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    return angle

# np.arctan2(c[1]-b[1], c[0]-b[0]) → angle of line BC
# np.arctan2(a[1]-b[1], a[0]-b[0]) → angle of line BA
# Then subtract one angle from the other → gives the angle between the two lines (BA and BC).


def get_distance(landmark_list):
    if len(landmark_list) < 2:
        return
    
    (x1, y1), (x2, y2) = landmark_list[0], landmark_list[1]
    L = np.hypot(x2-x1, y2-y1)          # calculates the Euclidean distance between two points.
    return np.interp(L, [0,1], [0, 1000])
#     That means if the actual distance L = 0.5,
#     then it becomes around 500 in the new scale.
