from dataset import CameraStream

import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsView,QPushButton
from PyQt5.QtGui import QPen, QColor, QPixmap, QImage, QBrush
from PyQt5.QtCore import Qt, QRectF, QTimer
import cv2

class MyView(QGraphicsView):
    def __init__(self,width,height):
        super().__init__()
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff) 
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFixedSize(width,height)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setSceneRect(QRectF(self.rect()))
        self.rect_item = None
        self.start_point = None
        self.end_point = None





    def setBackground(self, img):
        qimage = self.matToQImage(img)
        pixmap = QPixmap.fromImage(qimage)
        self.setBackgroundBrush(QBrush(pixmap))

    
    def matToQImage(self, mat):
        if mat.ndim == 2:
            qimage = QImage(mat.data, mat.shape[1], mat.shape[0], QImage.Format_Grayscale8)
        elif mat.ndim == 3:
            if mat.shape[2] == 3:
                qimage = QImage(mat.data, mat.shape[1], mat.shape[0], mat.shape[1] * 3, QImage.Format_RGB888)
            elif mat.shape[2] == 4:
                qimage = QImage(mat.data, mat.shape[1], mat.shape[0], mat.shape[1] * 4, QImage.Format_RGBA8888)
        return qimage

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_point = self.mapToScene(event.pos())
            self.end_point = self.start_point
            pen = QPen(QColor('yellow'))
            pen.setWidth(3)
            self.rect_item = self.scene.addRect(QRectF(self.start_point, self.end_point),pen)
            # print(self.start_point)

    def mouseMoveEvent(self, event):
        if self.rect_item is not None:
            self.end_point = self.mapToScene(event.pos())
            rect = QRectF(self.start_point, self.end_point).normalized()
            self.rect_item.setRect(rect)
            # print(self.end_point)
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.end_point = self.mapToScene(event.pos())
            rect = QRectF(self.start_point, self.end_point).normalized()
            self.rect_item.setRect(rect)
            self.rect_item = None
    def clear(self):
        # 清除所有已绘制的矩形
        for item in self.scene.items():
            self.scene.removeItem(item)
    def save(self):
        if(os.path.isdir('./position') == False):
            os.makedirs('./position')
        with open('./position/pos.txt','w') as file:
            for item in self.scene.items():
                file.write('x:{} y:{} width:{} height:{}'.format(item.rect().x(),item.rect().y(),item.rect().width(),item.rect().height()) + '\n')
        self.clear()
        



class CalibraMainWindow(QMainWindow):
    def __init__(self,cap = None,width=1920,height=1080):
        super().__init__()
        self.setFixedSize(width, height)
        # self.setGeometry(100,100,width,height)
        self.view = MyView(width,height)
        self.setCentralWidget(self.view)
        if(cap == None):
            self.cap = CameraStream(width=width,height=height)
            self.cap.start()
        else:
            self.cap = cap

        # 创建定时器，每隔10毫秒更新一次画面
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(10)

        # 添加清屏按钮
        clear_button = QPushButton("清屏", self)
        clear_button.clicked.connect(self.view.clear)
        clear_button.move(self.width() - clear_button.width()-20, self.height() - clear_button.height()-20)

        # 添加保存按钮
        save_button = QPushButton("保存", self)
        save_button.clicked.connect(self.view.save)
        save_button.move(clear_button.x(), clear_button.y() - save_button.height()-5)
    def update(self):
        ret, frame = self.cap.read()
        if ret:
            # 将OpenCV读取到的图像转换为Qt可识别的格式并设置为背景
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV读取的图像是BGR格式，需要转换为RGB格式
            self.view.setBackground(frame)
            self.view.viewport().update()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CalibraMainWindow()
    window.show()
    exit_status = app.exec_()
    window.cap.stop()
    exit(exit_status)
    

