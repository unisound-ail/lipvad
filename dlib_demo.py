#!/usr/bin/env python3
"""Visual VAD Demo
"""

from collections import deque

import cv2
import dlib

from helper import get_pose
from helper import mouth_length_width_ratio

def cv_show(title, img):
    """CVshow
    """
    cv2.imshow(title, img)
    if cv2.waitKey(7) & 0xFF == ord('q'):
        pass


def get_land_mark(shape_predictor, img, rect):
    """Return land mark coordinate as [(x1, y1), (x2, y2)...]
    """
    _s = shape_predictor(img, rect)
    return [(_s.part(i).x, _s.part(i).y) for i in range(0, _s.num_parts)]


def main():
    """Main
    """

    face_detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    _cap = cv2.VideoCapture(0)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    _out = cv2.VideoWriter('output.avi', fourcc, 16.0, (640, 480))
    write_video = False

    _total_diff, _last_ratio = 0.0, 0.0
    _diff_queue = deque(maxlen=3)
    _cnt_silence = 16
    while True:
        _, frame = _cap.read()

        cv_show("Origin", frame)

        rects, _, idx = face_detector.run(frame, 0)
        for i in idx:
            cv2.putText(frame, get_pose(i), (10 * i + 20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        _mouth_points = None
        _m_ratio = 0.0
        for _, _d in enumerate(rects):
            landmarks = get_land_mark(shape_predictor, frame, _d)

            if _mouth_points is None:
                _mouth_points = landmarks[60:68]
                _m_ratio = mouth_length_width_ratio(_mouth_points)

            for _lm in landmarks:
                cv2.circle(frame, (_lm[0], _lm[1]), 1, (0, 255, 0), -1)

            cv2.rectangle(frame, (_d.left(), _d.top()), (_d.right(), _d.bottom()), (255, 0, 0), 3)

        _total_score = _m_ratio + _last_ratio
        print("total_score:", _total_score)

        _diff = abs(_m_ratio ** 0.1 - _last_ratio ** 0.1)
        _diff_queue.append(_diff)
        _total_diff = sum(_diff_queue)
        print("total_diff:", _total_diff)

        _last_ratio = _m_ratio

        if _total_diff > 0.05 and _total_score > 0.12:
            _cnt_silence = 0
        else:
            _cnt_silence += 1

        if _cnt_silence < 8:
            cv2.putText(frame, "Speaking", (450, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        if write_video:
            _out.write(frame)
            if _cnt_silence > 60:
                break

        cv_show("Detected", frame)

    _cap.release()
    _out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()