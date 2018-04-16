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

def detection(lipvad, image, args):
    ratio = 0.0
    gray = lipvad.process(image)
    # detect faces and face pose
    rects, scores, indexs = lipvad.detect(gray)    
    for i, rect in enumerate(rects):
        shape = lipvad.predict(gray, rect)
        mouth = lipvad.extract_mouth(shape)
        #for (x, y) in mouth:
        #    cv2.circle(image, (x, y), 1, (0,0.255), -1) 
        ratio = lipvad.ratio(mouth)
        #cv2.putText(image, 'ratio: {}'.format(ratio), (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        # face pose
        pose = lipvad.pose(i)
        cv2.putText(image, 'face pose: {}'.format(pose), (20, 90),
             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        if args['log']: 
            print('face pose: {}'.format(pose))
        
    return image, ratio

def main(args):
    lipvad = LipVad(args, log=args['log'])
    lipvad.start()

    frames = []
    ratios = []

    cnt = 0
    while True:
        frame = lipvad.read()
        if frame is None:
            break

        cnt += 1
        frame, ratio = detection(lipvad, frame, args)
        frames.append(frame)
        ratios.append(ratio)
        #cv2.imshow('capture', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
        cv2.imwrite('{}.png'.format(cnt), frame)
        
    lipvad.stop()
    print('segments', lipvad.segments())
    print('FPS {}'.format(lipvad.fps()))
    print('time elapsed {}'.format(lipvad.elapsed()))

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
    ap.add_argument('-l', '--log', dest='log', action='store_true',
        help='logging on')
    args = vars(ap.parse_args())
    print(args)
    main(args)
