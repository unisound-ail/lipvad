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


class LipVad(object):

    MOUTH_MOVE_IDXS=(49, 61) # start from 1
    MOUTH_AR_THRESH_LOW = 0.1
    MOUTH_AR_THRESH_HIGH = 0.15
    MOUTH_AR_CONSEC_FRAMES = 4

    def __init__(self, args):
        self.args = args
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.args['shape_predictor'])
        self.vs = VideoStream(src=args['device'], usePiCamera=args['picamera'] > 0)
        #self.vs = VideoStream(src=args['video_filepath'], usePiCamera=args['picamera'] > 0)
        self.fps = FPS()

        self.indexs = None # face pose

        self._ratio = 0.0
        self.state = False
        self.cnt = 0
        self._start = []
        self._end = []

    def start(self):
        self.vs.start()
        self.fps.start()

    def stop(self):
        self.vs.stop()
        self.fps.stop()

    def read(self):
        frame = self.vs.read()
        self.fps.update()
        return frame

    def process(self, image):
        self._image = image
        #image = utils.resize(image, width=500)
        gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

    def detect(self, image):
        # detect faces and face pose
        rects, scores, self.indexs = self.detector.run(image, 1)    
        return rects, scores, self.indexs

    def predict(self, image, rect):
        shape = self.predictor(image, rect)
        shape = face_utils.shape_to_np(shape)
        return shape

    def pose(self, face_idx):
        return face_utils.FACIAL_POSE_IDXS[int(self.indexs[face_idx])]

    def extract_mouth(self, shape):
        (i, j) = LipVad.MOUTH_MOVE_IDXS
        return shape[i-1:j]

    def _mouth_aspect_ratio(self, mouth):
        # vertical 
        A = dist.euclidean(mouth[1], mouth[7])
        B = dist.euclidean(mouth[2], mouth[6])
        C = dist.euclidean(mouth[3], mouth[5])
    
        # Horizontal
        D = dist.euclidean(mouth[0], mouth[4])
    
        ratio = (A + B + C) / (3.0 * D)
        return ratio 

    def update(self, ratio):
        '''update vad state'''
        if ratio < LipVad.MOUTH_AR_THRESH_LOW:
            self.cnt += 1
            if self.state and self.cnt > LipVad.MOUTH_AR_CONSEC_FRAMES:
                self.cnt = 0
                print('end at frame {}'.format(self.fps.nframe()))
                cv2.putText(self._image, 'end', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                self.state = False
                self._end.append(self.fps.nframe())
        elif ratio >= LipVad.MOUTH_AR_THRESH_HIGH:
            if not self.state:
                self.state = True
                print('start at frame {}'.format(self.fps.nframe()))
                cv2.putText(self._image, 'start', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                self._start.append(self.fps.nframe())

    def ratio(self, image):
        self._ratio = self._mouth_aspect_ratio(image)
        self.update(self._ratio)

    def fps(self):
        return self.fps.fps()

    def elapsed(self):
        return self.fps.elapsed()

    def segments(self):
        return self._start, self._end

