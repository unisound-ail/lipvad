from threading import Thread
import sys
import cv2

if sys.version_info >= (3, 0):
    from queue import Queue
else:
    from Queue import Queue

class FileVideoStream:
    def __init__(self, path, queueSize=200):
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.Q = Queue(maxsize=queueSize)

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
