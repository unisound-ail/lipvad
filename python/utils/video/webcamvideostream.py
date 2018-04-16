from threading import Thread
import cv2
import sys

if sys.version_info >= (3, 0):
    from queue import Queue
else:
    from Queue import Queue

class WebcamVideoStream:
    def __init__(self, src=0, queuesize=128):
        self.stream = cv2.VideoCapture(src)
        self.stopped = False
        self.Q = Queue(maxsize=queuesize)

    def start(self):
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return

            if not self.Q.full():
                (grabbed, frame) = self.stream.read()
                if not grabbed:
                    self.stop()
                self.Q.put(frame)

    def read(self):
        return self.Q.get()

    def more(self):
        return self.Q.qsize() > 0

    def stop(self):
        self.stopped = True

    def qsize(self):
        return self.Q.qsize()
