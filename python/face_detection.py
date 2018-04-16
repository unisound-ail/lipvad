#!/usr/bin/env python

import utils
from utils.video import FileVideoStream, FPS, VideoStream
from utils import face_utils
import numpy as np
import datetime
import argparse
import time
import dlib
import cv2
from scipy.spatial import distance as dist
from lipvad import LipVad

def main(args):
    lipvad = LipVad(args)
    fps = FPS().start()
    lipvad.start()

    while True:
        ret, frame = lipvad.read()
        if frame is None:

        frame, ratio = lipvad.detection(frame, args)
        fps.update()
        cv2.imshow('capture', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    print('FPS {}'.format(fps.fps()))
    print('time elapsed {}'.format(fps.elapsed()))
    lipvad.stop()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--shape_predictor', required=None,
        default='shape_predictor_68_face_landmarks.dat',
        help='path to facial landmark predictor')
    ap.add_argument('-v', '--video_filepath', required=None,
        default='video.mpg',
        help='path to input video')
    ap.add_argument('-r', '--picamera', type=int, default=-1,
        help='whether or not the Raspberry Pi camera should be used')
    ap.add_argument('-d', '--device', type=int, default=0,
        help='camera device id')
    args = vars(ap.parse_args())
    main(args)
