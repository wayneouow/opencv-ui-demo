import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QGridLayout,
                             QLabel, QPushButton, QFileDialog)
from PyQt5.QtGui import QPixmap
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QFileDialog,QLabel,QWidget
import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import os
import torchvision
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchsummary import summary
import torch

def conv(image, kernel):
    image = np.pad (image, [(1, 1), (1, 1) ], mode= 'constant', constant_values=0)
    kernel_h, kernel_w = kernel.shape
    height, width = image.shape
    
    new_image = np.zeros ((height, width)).astype (np. float32)
    for y in range(0, height-2):
        for x in range(0, width-2) :
            new_image[y][x] = np.sum(image[y : y + kernel_h, x : x +kernel_w] * kernel).astype(np.float32)
    return new_image

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(611, 676)
        self.verticalLayoutWidget = QtWidgets.QWidget(Form)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(129, 30, 231, 631))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBox = QtWidgets.QGroupBox(self.verticalLayoutWidget)
        self.groupBox.setObjectName("groupBox")
        self.widget = QtWidgets.QWidget(self.groupBox)
        self.widget.setGeometry(QtCore.QRect(20, 20, 191, 171))
        self.widget.setObjectName("widget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.Btn11 = QtWidgets.QPushButton(self.widget)
        self.Btn11.setObjectName("Btn11")
        self.verticalLayout_2.addWidget(self.Btn11)
        self.Btn12 = QtWidgets.QPushButton(self.widget)
        self.Btn12.setObjectName("Btn12")
        self.verticalLayout_2.addWidget(self.Btn12)
        self.Btn13 = QtWidgets.QPushButton(self.widget)
        self.Btn13.setObjectName("Btn13")
        self.verticalLayout_2.addWidget(self.Btn13)
        self.verticalLayout.addWidget(self.groupBox)
        self.groupBox_3 = QtWidgets.QGroupBox(self.verticalLayoutWidget)
        self.groupBox_3.setObjectName("groupBox_3")
        self.layoutWidget = QtWidgets.QWidget(self.groupBox_3)
        self.layoutWidget.setGeometry(QtCore.QRect(20, 20, 191, 171))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.Btn21 = QtWidgets.QPushButton(self.layoutWidget)
        self.Btn21.setObjectName("Btn21")
        self.verticalLayout_3.addWidget(self.Btn21)
        self.Btn22 = QtWidgets.QPushButton(self.layoutWidget)
        self.Btn22.setObjectName("Btn22")
        self.verticalLayout_3.addWidget(self.Btn22)
        self.Btn23 = QtWidgets.QPushButton(self.layoutWidget)
        self.Btn23.setObjectName("Btn23")
        self.verticalLayout_3.addWidget(self.Btn23)
        self.verticalLayout.addWidget(self.groupBox_3)
        self.groupBox_2 = QtWidgets.QGroupBox(self.verticalLayoutWidget)
        self.groupBox_2.setObjectName("groupBox_2")
        self.layoutWidget_2 = QtWidgets.QWidget(self.groupBox_2)
        self.layoutWidget_2.setGeometry(QtCore.QRect(20, 20, 191, 171))
        self.layoutWidget_2.setObjectName("layoutWidget_2")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.layoutWidget_2)
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.Btn31 = QtWidgets.QPushButton(self.layoutWidget_2)
        self.Btn31.setObjectName("Btn31")
        self.verticalLayout_7.addWidget(self.Btn31)
        self.Btn32 = QtWidgets.QPushButton(self.layoutWidget_2)
        self.Btn32.setObjectName("Btn32")
        self.verticalLayout_7.addWidget(self.Btn32)
        self.Btn33 = QtWidgets.QPushButton(self.layoutWidget_2)
        self.Btn33.setObjectName("Btn33")
        self.verticalLayout_7.addWidget(self.Btn33)
        self.Btn34 = QtWidgets.QPushButton(self.layoutWidget_2)
        self.Btn34.setObjectName("Btn34")
        self.verticalLayout_7.addWidget(self.Btn34)
        self.verticalLayout.addWidget(self.groupBox_2)
        self.groupBox_4 = QtWidgets.QGroupBox(Form)
        self.groupBox_4.setGeometry(QtCore.QRect(370, 30, 229, 261))
        self.groupBox_4.setObjectName("groupBox_4")
        self.Btn4 = QtWidgets.QPushButton(self.groupBox_4)
        self.Btn4.setGeometry(QtCore.QRect(20, 220, 189, 23))
        self.Btn4.setObjectName("Btn4")
        self.gridLayoutWidget = QtWidgets.QWidget(self.groupBox_4)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(20, 30, 189, 181))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label_6 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 2, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 3, 0, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_7.setObjectName("label_7")
        self.gridLayout.addWidget(self.label_7, 2, 2, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 3, 2, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 2, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 1, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.lineEdit = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.lineEdit_4 = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.gridLayout.addWidget(self.lineEdit_4, 3, 1, 1, 1)
        self.gridLayout.addWidget(self.lineEdit_3, 2, 1, 1, 1)
        self.gridLayout.addWidget(self.lineEdit_2, 1, 1, 1, 1)
        self.gridLayout.addWidget(self.lineEdit, 0, 1, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_4.setText("")
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 1, 2, 1, 1)
        self.groupBox_5 = QtWidgets.QGroupBox(Form)
        self.groupBox_5.setGeometry(QtCore.QRect(370, 300, 229, 361))
        self.groupBox_5.setObjectName("groupBox_5")
        self.layoutWidget_3 = QtWidgets.QWidget(self.groupBox_5)
        self.layoutWidget_3.setGeometry(QtCore.QRect(20, 30, 191, 141))
        self.layoutWidget_3.setObjectName("layoutWidget_3")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.layoutWidget_3)
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.Btn5load = QtWidgets.QPushButton(self.layoutWidget_3)
        self.Btn5load.setObjectName("Btn5load")
        self.verticalLayout_6.addWidget(self.Btn5load)
        self.Btn51 = QtWidgets.QPushButton(self.layoutWidget_3)
        self.Btn51.setObjectName("Btn51")
        self.verticalLayout_6.addWidget(self.Btn51)
        self.Btn52 = QtWidgets.QPushButton(self.layoutWidget_3)
        self.Btn52.setObjectName("Btn52")
        self.verticalLayout_6.addWidget(self.Btn52)
        self.Btn53 = QtWidgets.QPushButton(self.layoutWidget_3)
        self.Btn53.setObjectName("Btn53")
        self.verticalLayout_6.addWidget(self.Btn53)
        self.Btn54 = QtWidgets.QPushButton(self.layoutWidget_3)
        self.Btn54.setObjectName("Btn54")
        self.verticalLayout_6.addWidget(self.Btn54)
        self.label_9 = QtWidgets.QLabel(self.groupBox_5)
        self.label_9.setGeometry(QtCore.QRect(50, 330, 80, 12))
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.groupBox_5)
        self.label_10.setGeometry(QtCore.QRect(60, 190, 128, 128))
        self.label_10.setAutoFillBackground(True)
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName("label_10")
        self.loadBtn1 = QtWidgets.QPushButton(Form)
        self.loadBtn1.setGeometry(QtCore.QRect(20, 270, 91, 61))
        self.loadBtn1.setObjectName("loadBtn1")
        self.loadBtn2 = QtWidgets.QPushButton(Form)
        self.loadBtn2.setGeometry(QtCore.QRect(20, 360, 91, 61))
        self.loadBtn2.setObjectName("loadBtn2")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        #######
        self.groupBox.setTitle(_translate("Form", "1. Image Processing"))
        ## 1-1
        self.Btn11.setText(_translate("Form", "1.1 Color Separation"))
        self.Btn11.clicked.connect(self.ColorSeparate)
        ## 1-2
        self.Btn12.setText(_translate("Form", "1.2 Color Transformation"))
        self.Btn12.clicked.connect(self.ColorTransform)
        ## 1-3
        self.Btn13.setText(_translate("Form", "1.3 Color Extraction"))
        self.Btn13.clicked.connect(self.ColorExtract)
        #######
        self.groupBox_3.setTitle(_translate("Form", "2. Image Smoothing"))
        ## 2-1
        self.Btn21.setText(_translate("Form", "2.1 Gaussian blur"))
        self.Btn21.clicked.connect(self.GaussianBlur)
        ## 2-2
        self.Btn22.setText(_translate("Form", "2.2 Bilateral filter"))
        self.Btn22.clicked.connect(self.BilateralFilter)
        ## 2-3
        self.Btn23.setText(_translate("Form", "2.3 Median filter"))
        self.Btn23.clicked.connect(self.MedianBlur)
        #######
        self.groupBox_2.setTitle(_translate("Form", "3. Edge Detection"))
        ## 3-1
        self.Btn31.setText(_translate("Form", "3.1 Sobel X"))
        self.Btn31.clicked.connect(self.Sobelx)
        ## 3-2
        self.Btn32.setText(_translate("Form", "3.2 Sobel Y"))
        self.Btn32.clicked.connect(self.Sobely)
        ##3-3
        self.Btn33.setText(_translate("Form", "3.3 Combination and Threshold"))
        self.Btn33.clicked.connect(self.SobelCombination)
        ## 3-4
        self.Btn34.setText(_translate("Form", "3.4 Gradient Angle"))
        self.Btn34.clicked.connect(self.GradientAngle)
        #######
        self.groupBox_4.setTitle(_translate("Form", "4. Transforms"))
        ## 4
        self.Btn4.setText(_translate("Form", "4. Transforms"))
        self.Btn4.clicked.connect(self.Transforms)
        #
        self.label_6.setText(_translate("Form", "Tx:"))
        self.label_5.setText(_translate("Form", "Ty:"))
        self.label_7.setText(_translate("Form", "pixel"))
        self.label_8.setText(_translate("Form", "pixel"))
        self.label_2.setText(_translate("Form", "deg"))
        self.label_3.setText(_translate("Form", "Scaling:"))
        self.label.setText(_translate("Form", "Rotation:"))
        #######
        self.groupBox_5.setTitle(_translate("Form", "5. VGG19"))
        ## 5load
        self.Btn5load.setText(_translate("Form", "Load Image"))
        self.Btn5load.clicked.connect(self.LoadPic)
        ## 5-1
        self.Btn51.setText(_translate("Form", "5.1 Show Augmented Image"))
        self.Btn51.clicked.connect(self.ShowAugmentImage) 
        ## 5-2
        self.Btn52.setText(_translate("Form", "5.2 Show Model Structure"))
        self.Btn52.clicked.connect(self.ShowModel) 
        ## 5-3
        self.Btn53.setText(_translate("Form", "5.3 Show Acc and Loss"))
        self.Btn53.clicked.connect(self.ShowFigure) 
        ## 5-4
        self.Btn54.setText(_translate("Form", "5.4 Inference"))
        self.Btn54.clicked.connect(self.Inference) 
        self.label_9.setText(_translate("Form", "Predict = "))
        self.label_10.setText(_translate("Form", "Inference Image"))

        self.loadBtn1.setText(_translate("Form", "Load\nImage 1"))
        self.loadBtn1.clicked.connect(self.LoadImg1)
        self.loadBtn2.setText(_translate("Form", "Load\nImage 2"))
        self.loadBtn2.clicked.connect(self.LoadImg2)

    def ColorSeparate(self):
        cv2.destroyAllWindows()
        file, _ = QtWidgets.QFileDialog.getOpenFileName(filter='Image Files (*.png *.jpg *.jpeg *.bmp)')
        if "file" in locals():
            img = cv2.imread(file)
            b,g,r = cv2.split(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            z = np.zeros_like(img)
            b = cv2.merge([b,z,z])
            g = cv2.merge([z,g,z])
            r = cv2.merge([z,z,r])
            cv2.imshow("blue",b)
            cv2.imshow("green",g)
            cv2.imshow("red",r)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def ColorTransform(self):
        cv2.destroyAllWindows()
        file, _ = QtWidgets.QFileDialog.getOpenFileName(filter='Image Files (*.png *.jpg *.jpeg *.bmp)')
        if "file" in locals():
            img = cv2.imread(file)
            gray_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            b,g,r = cv2.split(img)  
            gray_avg = cv2.addWeighted(b, 0.333, g, 0.333, 0)
            gray_avg = cv2.addWeighted(gray_avg, 0.667, r, 0.333, 0)
            cv2.imshow("gray_cvt",gray_cvt)
            cv2.imshow("gray_avg",gray_avg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def ColorExtract(self):
        cv2.destroyAllWindows()
        file, _ = QtWidgets.QFileDialog.getOpenFileName(filter='Image Files (*.png *.jpg *.jpeg *.bmp)')
        if "file" in locals():
            img = cv2.imread(file)
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_img, (20, 25, 25), (85, 255,255))
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            ex_img = cv2.bitwise_not(mask_bgr, img, mask=mask)
            cv2.imshow("mask",mask)
            cv2.imshow("ex",ex_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
    def GaussianBlur(self):
        cv2.destroyAllWindows()
        if "filename1" in globals():
            img = cv2.imread(filename1)
            cv2.imshow('GaussianBlur', img)   
            def fn(val):
                blur = cv2.GaussianBlur(img, (2*val+1, 2*val+1), 0)
                cv2.imshow('GaussianBlur', blur)  
            cv2.createTrackbar('m', 'GaussianBlur',0, 5, fn)
            keycode = cv2.waitKey(0)
            cv2.destroyAllWindows()

    def BilateralFilter(self):
        cv2.destroyAllWindows()
        if "filename1" in globals():
            img = cv2.imread(filename1)
            cv2.imshow('BilateralFilter', img)
            def fn(val):
                bilateral = cv2.bilateralFilter(img, 2*val+1, 90, 90)
                cv2.imshow('BilateralFilter', bilateral )
            cv2.createTrackbar('m', 'BilateralFilter',0, 5, fn)
            keycode = cv2.waitKey(0)
            cv2.destroyAllWindows()

    def MedianBlur(self):
        cv2.destroyAllWindows()
        if "filename1" in globals():
            img = cv2.imread(filename1)
            cv2.imshow('MedianBlur', img)
            def fn(val):
                blur = cv2.medianBlur(img, 2*val+1)
                cv2.imshow('MedianBlur', blur )    
            cv2.createTrackbar('m', 'MedianBlur',0, 5, fn)
            keycode = cv2.waitKey(0)
            cv2.destroyAllWindows()

    
    def Sobelx(self):
        global fileSobel
        fileSobel, _ = QtWidgets.QFileDialog.getOpenFileName(filter='Image Files (*.png *.jpg *.jpeg *.bmp)')
        if "fileSobel" in globals():
            img = cv2.imread(fileSobel)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3,3), 0)
            xfilter = np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]])
            height, width = gray.shape[:2]  
            global sobel_x, sobel_x_s
            sobel_x = np.zeros_like(gray)
            sobel_x_s = np.zeros_like(gray)
            for y in range(1, height-1):
                for x in range(1, width-1):
                    conv_val = np.sum(gray[y-1 : y+2, x-1 : x+2] * xfilter)
                    sobel_x[y, x] = np.abs(conv_val)
                    sobel_x_s[y, x] = conv_val
            cv2.imwrite("output/sobel_x.jpg", sobel_x)
            cv2.imshow("sobelX", sobel_x)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    def Sobely(self):
        #fileSobel, _ = QtWidgets.QFileDialog.getOpenFileName(filter='Image Files (*.png *.jpg *.jpeg *.bmp)')
        if "fileSobel" in globals():
            img = cv2.imread(fileSobel)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3,3), 0)
            yfilter = np.array([[-1,-2,-1],
                                [ 0, 0, 0],
                                [ 1, 2, 1]])
            height, width = gray.shape[:2]
            global sobel_y, sobel_y_s
            sobel_y = np.zeros_like(gray)
            sobel_y_s = np.zeros_like(gray)
            for y in range(1, height-1):
                for x in range(1, width-1):
                    conv_val = np.sum(gray[y-1 : y+2, x-1 : x+2] * yfilter)
                    sobel_y[y, x] = np.abs(conv_val)
                    sobel_y_s[y, x] = conv_val

            cv2.imwrite("output/sobel_y.jpg", sobel_y)
            cv2.imshow("sobelY", sobel_y)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def SobelCombination(self):
        if("sobel_x" in globals() and "sobel_y" in globals()):
            sx = sobel_x
            sy = sobel_y
            global com
            com = np.zeros_like(sx)
            height, width = sx.shape[:2]
            for j in range(1, height-1):
                for i in range(1, width-1):
                    com[j,i] = np.sqrt(sx[j,i]**2 + sy[j,i]**2)
            com = (com * 255.0 / com.max()).astype(np.uint8)
            ret, th = cv2.threshold(com, 128, 255, cv2.THRESH_BINARY)
            cv2.imwrite("output/sobel_combination.jpg", com)
            cv2.imshow('combination', com)
            cv2.imshow('th', th)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("NO sobel x or sobel y")

    def GradientAngle(self):
        if "fileSobel" in globals():
        #if("sobel_x_s" in globals() and "sobel_y_s" in globals() and "com" in globals()):
            # gX = sobel_x_s
            # gY = sobel_y_s
            img = cv2.imread(fileSobel)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3,3), 0)
            # gX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
            # gY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
            xfilter = np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]])
            yfilter = np.array([[-1,-2,-1],
                                [ 0, 0, 0],
                                [ 1, 2, 1]])
            gX = conv(gray, xfilter)
            gY = conv(gray, yfilter)
            

            height, width = sobel_x_s.shape[:2]
            mask1 = np.zeros_like(gray).astype(np.int8)
            mask2 = np.zeros_like(gray).astype(np.int8)
            t = np.zeros_like(gray).astype(np.float32)
            
            for j in range(1, height-1):
                for i in range(1, width-1):
                    t[j, i] = np.arctan2(gY[j, i], gX[j, i])*(180/np.pi) + 180
                    if(t[j, i] >= 120 and t[j, i] <= 180):
                        mask1[j, i] = 255
                    else:
                        mask1[j, i] = 0.
                    if(t[j, i] >= 210 and t[j, i] <= 330):
                        mask2[j, i] = 255
                    else:
                        mask2[j, i] = 0
        
            mask1 = mask1.astype(np.uint8)
            mask2 = mask2.astype(np.uint8)
            combination = com.astype(np.uint8)
            g1 = cv2.bitwise_and(combination, mask1)
            g2 = cv2.bitwise_and(combination, mask2)

            # cv2.imshow('comb', combination)
            # cv2.imshow('mask1', mask1)
            # cv2.imshow('mask2', mask2)
            cv2.imshow('120˚~180˚', g1)
            cv2.imshow('210˚~330˚', g2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def Transforms(self):
        cv2.destroyAllWindows()
        file, _ = QtWidgets.QFileDialog.getOpenFileName(filter='Image Files (*.png *.jpg *.jpeg *.bmp)')
        if "file" in locals():
            img = cv2.imread(file)
            angle = float(self.lineEdit.text()) if self.lineEdit.text() else 0.
            scale = float(self.lineEdit_2.text()) if self.lineEdit_2.text() else 1.
            tx = float(self.lineEdit_3.text()) if self.lineEdit_3.text() else 0.
            ty = float(self.lineEdit_4.text()) if self.lineEdit_4.text() else 0.
            R = cv2.getRotationMatrix2D((240, 200), angle, scale)
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            img = cv2.warpAffine(img, R,(img.shape[1], img.shape[0]))
            img = cv2.warpAffine(img, M,(img.shape[1], img.shape[0]))
            
            cv2.imshow('b', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def LoadImg1(self):
        global filename1
        filename1, _ = QtWidgets.QFileDialog.getOpenFileName(filter='Image Files (*.png *.jpg *.jpeg *.bmp)')
    
    def LoadImg2(self):
        global filename2
        filename2, _ = QtWidgets.QFileDialog.getOpenFileName(filter='Image Files (*.png *.jpg *.jpeg *.bmp)')

    def LoadPic(self):
        global filename
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(filter='Image Files (*.png *.jpg *.jpeg *.bmp)')

        if not filename:
            return
        pic = QPixmap(filename)
        if pic.isNull():
            return
        pic_resize = pic.scaled(128, 128)
        self.label_10.setPixmap(pic_resize)

    def ShowAugmentImage(self):
        folder_path = QFileDialog.getExistingDirectory()
        #folder_path = 'Q5_image\Q5_1'
        file_list = os.listdir(folder_path)[:9]

        rows = 3
        columns = 3
        fig, axs = plt.subplots(rows, columns, figsize=(8, 8))

        for i in range(rows):
            for j in range(columns):
                image_path = os.path.join(folder_path, file_list[i * columns + j])

                img = Image.open(image_path)
                size = 100
                transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.6),
                    transforms.RandomVerticalFlip(p=0.6),
                    transforms.RandomRotation(30)
                ])
                new_img = transform(img)
                
                axs[i, j].imshow(new_img)
        plt.show()
    
    def ShowModel(self):
        model = torchvision.models.vgg19_bn(num_classes=10)
        print(summary(model, (3, 32,32)))

    def ShowFigure(self):
        img = Image.open("LossAndAccuracy.png")
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.show()

    def Inference(self):
        if "filename" in globals():
            image_path = filename
            img = Image.open(image_path)
            myModel = torchvision.models.vgg19_bn(num_classes=10)
            myModel.load_state_dict(torch.load("weight.pth"))
            myModel.eval()
            preprocess = transforms.Compose([
                        transforms.ToTensor(),
                    ])
            input_image = Image.open(image_path)
            input_image = input_image
            input_tensor = preprocess(input_image)
            input_batch = input_tensor.unsqueeze(0)
            with torch.no_grad():
                output = myModel(input_batch)
            _, predicted_idx = torch.max(output, 1)
            class_labels = ['plane', 'automobile', 'bird', 'cat',
                    'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            predicted_label = class_labels[predicted_idx.item()].strip()
            print("Predicted label:", predicted_idx, predicted_label)
            result = "Predict = " + str(predicted_label)
            self.label_9.setText(result)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0] * 100
            
            plt.figure(figsize=(10, 5))
            plt.bar(class_labels, probabilities)
            plt.xlabel('Classes')
            plt.ylabel('Probabilities')
            plt.title('Probability of Each Class')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
    
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
