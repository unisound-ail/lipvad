from collections import OrderedDict

from scipy.spatial import distance as dist

# 5 HOG filters. A front looking, left looking, right looking, 
# front looking but rotated left, and finally a front looking but rotated right one
FACIAL_POSE_IDXS = OrderedDict([
    (0, 'front_looking'),
    (1, 'left_looking'),
    (2, 'right_looking'),
    (3, 'front_looking_rotated_left'),
    (4, 'front_looking_rotated_right'),
])


def get_pose(idx):
    return FACIAL_POSE_IDXS[idx]


def mouth_length_width_ratio(mouth):
    # vertical 
    A = dist.euclidean(mouth[1], mouth[7])
    B = dist.euclidean(mouth[2], mouth[6])
    C = dist.euclidean(mouth[3], mouth[5])
    
    # Horizontal
    D = dist.euclidean(mouth[0], mouth[4])

    ratio = (A + B + C) / (3.0 * D)
    return ratio
