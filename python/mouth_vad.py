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


MOUTH_MOVE_IDXS=(61, 68) # start from 1
MOUTH_AR_THRESH = 0.1
MOUTH_AR_CONSEC_FRAMES = 4

def mouth_aspect_ratio(mouth):
    # vertical 
    A = dist.euclidean(mouth[1], mouth[7])
    B = dist.euclidean(mouth[2], mouth[6])
    C = dist.euclidean(mouth[3], mouth[5])

    # Horizontal
    D = dist.euclidean(mouth[0], mouth[4])

    mouth = (A + B + C) / (3.0 * D)
    return mouth

def detection(image, args):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args['shape_predictor'])

    #image = utils.resize(image, width=500)
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = image

    ratio = 0.0
    # detect faces and face pose
    rects, scores, indexs = detector.run(gray, 1)    
    for i, rect in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        (mi, mj) = MOUTH_MOVE_IDXS
        mouth = shape[mi-1:mj+1]
        #for (x, y) in shape:
        #    cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

        # mouth aspect ratio for vad
        ratio = mouth_aspect_ratio(mouth)
        print('ratio', ratio)
        #cv2.putText(image, 'ratio: {}'.format(ratio), (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
        # face pose
        cv2.putText(image, 'face pose: {}'.format(face_utils.FACIAL_POSE_IDXS[int(indexs[i])]), (20, 90),
             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
    return image, ratio

def main(args):
    vs = VideoStream(src=args['device'], usePiCamera=args['picamera'] > 0).start()
    #vs = FileVideoStream(args['video_filepath']).start()
    time.sleep(2.0)
    fps = FPS().start()

    frames = []
    ratios = []

    state = False
    cnt = 0
    
    start = []
    end = []
    while True:
        frame = vs.read()
        if frame is None:
            break

        frame, ratio = detection(frame, args)
        frames.append(frame)
        ratios.append(ratio)

        fps.update()

        if ratio < MOUTH_AR_THRESH:
            cnt += 1
            if state and cnt > MOUTH_AR_CONSEC_FRAMES:
                cnt = 0
                print('end at frame {}'.format(fps.nframe()))
                cv2.putText(frame, 'end', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                state = False
                end.append(fps.nframe())
        elif ratio >= 0.15:
            if not state:
                state = True
                print('start at frame {}'.format(fps.nframe()))
                cv2.putText(frame, 'start', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                start.append(fps.nframe())

        cv2.imshow('capture', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
        
    print('start', start)
    print('end', end)

    #print('ratios', ratios)
    #plt.plot(ratios)
    #plt.savefig('mouth_ratios.png')

    #frames = np.stack(frames)
    #print(frames.shape)
    #skvideo.io.vwrite('output', frames)
    #for i, frame in enumerate(frames):
    #    cv2.imwrite('%d.png' % i, frame) 

    #imageio.mimsave('output.gif', frames)

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
