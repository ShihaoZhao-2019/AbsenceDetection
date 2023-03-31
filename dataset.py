import cv2
import threading
import copy
import time

class CameraStream(threading.Thread):
    def __init__(self, src=0,width=1920,height=1080,fps=60):
        threading.Thread.__init__(self)
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # 设置帧宽度
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)  # 设置帧高度
        self.stream.set(cv2.CAP_PROP_FPS, fps)  # 设置帧率
        self.ret1 = None
        self.frame1 = None
        self.ret2 = None
        self.frame2 = None
        self.stopped = False
        self.read_flag = 1
        self.flag_lock = threading.Lock()

        # fps
        self.start_time = 0

    def start(self):
        threading.Thread.start(self)
        return self

    def run(self):
        
        while not self.stopped:
            if(self.read_flag == 2):
                ret, frame = self.stream.read()

                self.ret1 = copy.deepcopy(ret)

                self.frame1 = copy.deepcopy(frame)
                with self.flag_lock:
                    self.read_flag = 1
            elif(self.read_flag == 1):
                ret, frame = self.stream.read()

                self.ret2 = copy.deepcopy(ret)

                self.frame2 = copy.deepcopy(frame)
                with self.flag_lock:
                    self.read_flag = 2


    def stop(self):
        self.stopped = True
        self.stream.release()

    def read(self):
        with self.flag_lock:
            if(self.read_flag == 1):
                ret = self.ret1
                frame = self.frame1
            elif(self.read_flag == 2):
                ret = self.ret2
                frame = self.frame2
        # print("self.read_flag:",self.read_flag)
        return ret, frame
