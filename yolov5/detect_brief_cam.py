import cv2
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import scale_coords,non_max_suppression
from utils.plots import Annotator,colors
import numpy as np
import torch
import threading
from copy import deepcopy



# 摄像头or视频
camera_id = 0
img_height = 1080
img_width = 1920
thread_lock = threading.Lock()
thread_exit = False

class myThread(threading.Thread):
    def __init__(self, camera_id, img_height, img_width):
        super(myThread, self).__init__()
        self.camera_id = camera_id
        self.img_height = img_height
        self.img_width = img_width
        self.frame = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(3,img_width) #设置分辨率
        self.cap.set(4,img_height)
    def get_frame(self):
        return deepcopy(self.frame)

    def run(self):
        global thread_exit

        while not thread_exit:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (self.img_width, self.img_height))
                thread_lock.acquire()
                self.frame = frame
                thread_lock.release()
            else:
                thread_exit = True
        self.cap.release()

thread = myThread(camera_id, img_height, img_width)
thread.start()

# yolo检测设置
conf_thres = 0.5
iou_thres = 0.45
line_thickness=3  # bounding box thickness (pixels)

# 加载模型
cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')
# model_weight_path = "yolov5x6.pt"
model_weight_path = "yolov5n.pt"
model = DetectMultiBackend(model_weight_path, device=device, dnn=False)
if device.type != 'cpu':
    model(torch.zeros(1, 3, 640, 640).to(device).type_as(next(model.model.parameters())))  # warmup


# 定位模块
fy = 1120.74894580751
fx = 1121.53090010034
u0 = 974.549512973486
v0 = 518.718188662991
w_zsh = 450
h_zsh = 1655

def caulatePosition(box_xyxy):
    w = box_xyxy[2] - box_xyxy[0]
    h = box_xyxy[3] - box_xyxy[1]
    centerPoint = (box_xyxy[0] + w / 2, box_xyxy[1] + h / 2)
    distance = ((w_zsh * h_zsh * fy * fx) / (w * h)) ** 0.5
    # distance = (w_zsh*fx)/w
    # distance = (h_zsh * fy) / h
    xw = (distance * (centerPoint[0] - u0)) /  fx
    yw = (distance * (centerPoint[1] - v0)) /  fy
    return xw,distance, yw

# 处理照片 检测+定位
def caulateFrame(img0):
    img = letterbox(img0, [640, 640], stride=64, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float()
    img /= 255
    img = img[None]  # expand for batch dim

    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=1000)
    names = model.names

    for det in pred:
        if len(det):

            annotator = Annotator(img0, line_width=line_thickness, example=str(names))
            if len(det):
                imgwh = img.shape[2:]
                img0shape = img0.shape
                detbbox = det[:, :4]

                det[:, :4] = scale_coords(imgwh, detbbox, img0shape).round()
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    if c == 0:
                        position = caulatePosition(xyxy)
                        position = "x:" + str(int(position[0])/1000) + " y:" + str(int(position[1])/1000) + " z:" + str(int(position[2])/1000)
                        label = f'{names[c]} {conf:.2f} {position}'
                    else:
                        # label = f'{names[c]} {conf:.2f}'
                        continue
                    annotator.box_label(xyxy, label, color=colors(c, True))
            # Stream results
            img0 = annotator.result()
    return img0

def myMain():
    global thread_exit
    while not thread_exit:
        thread_lock.acquire()
        img0 = thread.get_frame()
        thread_lock.release()
        img0 = caulateFrame(img0)
        cv2.imshow(model_weight_path, img0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            thread_exit = True
    thread.join()


if __name__ == '__main__':
    # get_Video()
    myMain()