import numpy as np
import os
import sys
import cv2

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

def is_cv2():
    return check_opencv_version('2.')

def is_cv3():
    return check_opencv_version('3.')

def check_opencv_version(major, lib=None):
    if lib is None:
        import cv2 as lib
    return lib.__version__.startswith(major)
