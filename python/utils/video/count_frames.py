from ..convenience import is_cv3
import cv2

def count_frames(path, override=False):
    video = cv2.VideoCapture(path)
    total = 0
 
    if override:
        total = count_frames_manual(video)
    else:
        try:
            if is_cv3():
                total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            else:
                total = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        except:
            total = count_frames_manual(video)

    video.release()
    return total

def count_frames_manual(video):
    total = 0
    
    while True:
        (grabbed, frame) = video.read()

        if not grabbed:
            break

        total += 1

    return total
