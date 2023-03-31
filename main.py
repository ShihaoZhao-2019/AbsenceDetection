import os
import sys
import cv2
import time
from PyQt5.QtWidgets import QApplication
from dataset import CameraStream
from detection import Detection
from calibration import CalibraMainWindow

class Task():
    def __init__(self) -> None:
        pass
        self.app = QApplication(sys.argv)
        self.model = Detection()
        self.cap = CameraStream()
        self.cap.start()
        self.cali_window = CalibraMainWindow(self.cap)
        self.iou_threshold = 0

    def read_box(self):
        """
        return x:y:w:h
        """
        with open('position/pos.txt', 'r') as f:
            lines = f.read()
        lines = lines.split('\n')
        boxes = [line.split(' ') for line in lines]
        boxes.pop()
        for box in boxes:
            if(len(box) ==  4):
                box[0] = int(float(box[0].split(':')[1]))
                box[1] = int(float(box[1].split(':')[1]))
                box[2] = int(float(box[2].split(':')[1]))
                box[3] = int(float(box[3].split(':')[1]))

                self.xywh_to_xyxy(box)

        return boxes
    def xywh_to_xyxy(self,xywh):
        """
        xywh to xyxy
        """
        x1, y1, w, h = xywh[0],xywh[1],xywh[2],xywh[3]
        xywh[0] = x1
        xywh[1] = y1
        xywh[2] = x1+w
        xywh[3] = y1+h
        return x1, y1, x1+w, y1+h

    def bbox_iou(self,box1, box2):
        # 计算两个矩形的面积
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # 计算两个矩形相交部分的坐标
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # 计算相交矩形的面积
        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        # 计算并集矩形的面积
        union = area1 + area2 - intersection

        # 计算IoU
        iou = intersection / union

        return iou

    def draw_coordinates(self,coordinates, image):
        """
        将一组坐标绘制到给定的图像上，并在每个方框内部绘制美美的图案
        
        参数:
        coordinates: 一个包含坐标的元组列表，每个元组以(x1, y1, x2, y2)的形式表示
        image: 要绘制坐标的图像
        
        返回值:
        绘制好坐标的图像
        """
        # 在图像上绘制每个坐标
        for coordinate in coordinates:
            x1, y1, x2, y2 = coordinate
            
            # 绘制方框
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 在方框内部绘制美美的图案
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            line_length = int(min(x2 - x1, y2 - y1) / 2)
            thickness = int(line_length / 8)
            
            # 绘制圆形
            cv2.circle(image, (center_x, center_y), line_length, (0, 0, 255), thickness)
            
            # 绘制十字
            cv2.line(image, (center_x - line_length, center_y), 
                    (center_x + line_length, center_y), (0, 0, 255), thickness)
            cv2.line(image, (center_x, center_y - line_length), 
                    (center_x, center_y + line_length), (0, 0, 255), thickness)
        
        return image


    def caluate_abensce(self,pred):
        absense_seat_list = []
        for seat_box in self.seat_boxs:
            absense_flag = False
            for pred_box in pred:
                if self.bbox_iou(seat_box, pred_box) > self.iou_threshold:
                    absense_flag = True
                    break
            if absense_flag == False:
                absense_seat_list.append(seat_box)
                
        return absense_seat_list


    
    def run(self):
        # 启动之后首先进行位置标定
        self.cali_window.show()
        self.app.exec_()
        del self.cali_window

        # 读取岗位信息
        self.seat_boxs = self.read_box()

        # 进行目标检测
        start_time = time.time()
        while True:
            ret, frame = self.cap.read()
            if(ret):
                frame,dst = self.model(frame)
                # 对检测结果进行处理判断缺岗位
                abensce_lsit = self.caluate_abensce(dst)
                frame = self.draw_coordinates(abensce_lsit,frame)
                cv2.imshow("缺岗检测",frame)
                print("show fps:",1/(time.time()-start_time))
                # print('len:',len(dst))
                start_time = time.time()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.stop()
        cv2.destroyAllWindows()


        # 标定之后进行实时检测
        # exit(exit_status)
t = Task()
t.run()


    
