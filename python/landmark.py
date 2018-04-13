#!/usr/bin/env python

#import matplotlib as mpl 
#mpl.use('AGG')
#import matplotlib.pyplot as plt
import utils
from utils.video import FileVideoStream, FPS, VideoStream
from utils import face_utils
import numpy as np
import datetime
import argparse
import time
import dlib
import cv2
#import skvideo.io
from scipy.spatial import distance as dist
#import imageio


def detection(image, args):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args['shape_predictor'])

    #image = utils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray = image

    ratio = 0.0
    # detect faces and face pose
    rects, scores, indexs = detector.run(gray, 1)    
    for i, rect in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        mouth = shape
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
       
    return image, ratio

def main(args):
    vs = VideoStream(src=args['device'], usePiCamera=args['picamera'] > 0).start()
    fps = FPS().start()

    frames = []

    state = False
    cnt = 0
    
    while True:
        frame = vs.read()
        if frame is None:
            break

        frame, ratio = detection(frame, args)

        fps.update()

        cv2.imshow('capture', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    fps.stop()
    print('FPS {}'.format(fps.fps()))
    print('time elapsed {}'.format(fps.elapsed()))
    vs.stop()
    cv2.destroyAllWindows()

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
