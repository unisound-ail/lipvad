from collections import OrderedDict
import numpy as np
import cv2

FACIAL_LANDMARKS_IDXS = OrderedDict([
    ('mouth', (48, 68)),
    ('right_eyebrow', (17, 22)),
    ('left_eyebrow', (22, 27)),
    ('right_eye', (36, 42)),
    ('left_eye', (42, 48)),
    ('nose', (27, 36)),
    ('jaw', (0, 17))
])

# 5 HOG filters. A front looking, left looking, right looking, 
# front looking but rotated left, and finally a front looking but rotated right one
FACIAL_POSE_IDXS = OrderedDict([
    (0, 'front_looking'),
    (1, 'left_looking'),
    (2, 'right_looking'),
    (3, 'front_looking_rotated_left'),
    (4, 'front_looking_rotated_right'),
])

def rect_to_bb(rect):
    # take a bounding predicted by dlib and conver it 
    # to the format(x, y, w, h) as we would normally do 
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
  
    return (x, y, w, h)


def shape_to_np(shape, dtype='int'):
    # initalize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords

def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
    overlay = image.copy()
    output = image.copy()

    if colors is None:
        # colors for mouth, left_eyebrow, right_eyebrow, right_eye, left_eye, nose, jaw
        colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
                  (168, 100, 168), (158, 163, 32),
                  (163, 38, 32), (180, 42, 220)]

    for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
        (j, k) = FACIAL_LANDMARKS_IDXS[name]
        pts = shape[j:k]

        if name == 'jaw':
            for l in range(1, len(pts)):
                ptA = tuple(pts[l - 1])
                ptB = tuple(pts[l])
                cv2.line(overlay, ptA, ptB, colors[i], 2)
        else:
            hull = cv2.convexHull(pts)
            cv2.drawContours(overlay, [hull], -1, colors[i], -1)

    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    return output
