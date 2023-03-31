import sys
sys.path.append("./yolov5")
import cv2
import time
import numpy as np
import torch
import threading
from copy import deepcopy

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import scale_coords,non_max_suppression
from utils.plots import Annotator,colors
from dataset import CameraStream


class Detection():
    def __init__(self,weights='yolov5/yolov5x6.pt',conf_thres=0.45,iou_thres=0.1) -> None:
        #initq
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')
        self.model_weight_path = weights
        self.model = DetectMultiBackend(self.model_weight_path, device=self.device, dnn=False)
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, 640, 640).to(self.device).type_as(next(self.model.model.parameters())))  # warmup

    def caulateFrame(self,img0):
        img = letterbox(img0, [640, 640], stride=64, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device).float()
        img /= 255
        img = img[None]  # expand for batch dim

        pred = self.model(img, augment=False, visualize=False)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, None, False, max_det=1000)
        names = self.model.names

        person_boxs = []
        for det in pred:
            if len(det):
                annotator = Annotator(img0, line_width=3, example=str(names))
                if len(det):
                    imgwh = img.shape[2:]
                    img0shape = img0.shape
                    detbbox = det[:, :4]

                    det[:, :4] = scale_coords(imgwh, detbbox, img0shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        if c == 0:
                            label = f'{names[c]} {conf:.2f}'
                            person_boxs.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), conf, c])
                        else:
                            continue
                        annotator.box_label(xyxy, label, color=colors(c, True))
                # Stream results
                img0 = annotator.result()
        return img0,person_boxs
    def __call__(self, img):
        return self.caulateFrame(img)

       

  
if __name__ == "__main__":
    detector = Detection()
    cam = CameraStream()
    cam.start()
    start_time = time.time()
    while True:
        ret, frame = cam.read()
        if(ret):
            frame,dst = detector(frame)
            cv2.imshow("demo",frame)
            
            print("show fps:",1/(time.time()-start_time))
            print('len:',len(dst))
            start_time = time.time()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.stop()
    cv2.destroyAllWindows()
    
