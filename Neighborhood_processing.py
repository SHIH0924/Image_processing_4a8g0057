import cv2 as cv
from tkinter import messagebox as msgbox
import im
def nothing(x):
    pass
def Mean_Filter():# 均值濾波器函式
    try:
        cv.namedWindow('image')# 新建視窗
        cv.createTrackbar('kernel','image',3,20,nothing)
        cv.createTrackbar('kernel1','image',3,20,nothing)
        while(1):
            r = cv.getTrackbarPos('kernel','image')
            r1 = cv.getTrackbarPos('kernel1','image')
            blur = cv.blur(im.im, (r, r1))# 設定均值化圖片及內核大小
            cv.imshow("image", blur)# 不斷更新當前圖片
            # 如按下ENTER將圖片匯入"影像處理程式開發平台"視窗
            k = cv.waitKey(1) & 0xFF
            if k == 13:
                cv.destroyWindow('image')
                return blur
            elif k==32:
                cv.destroyWindow('image')
                return im.im
    except Exception:
        msgbox.showerror("Error", "Mean Filter error!!!")

def Box_Filter():# 方框濾波器函式
    try:
        cv.namedWindow('image')# 新建視窗
        cv.createTrackbar('kernel','image',3,20,nothing)
        cv.createTrackbar('kernel1','image',3,20,nothing)
        while(1):
            r = cv.getTrackbarPos('kernel','image')
            r1 = cv.getTrackbarPos('kernel1','image')
            box = cv.boxFilter(im.im, -1, (r, r1), normalize=True)
            # 設定方框化圖片及內核大小
            cv.imshow("image", box)# 不斷更新當前圖片
            # 如按下ENTER將圖片匯入"影像處理程式開發平台"視窗
            k = cv.waitKey(1) & 0xFF
            if k == 13:
                cv.destroyWindow('image')
                return box
            elif k==32:
                cv.destroyWindow('image')
                return im.im
    except Exception:
        msgbox.showerror("Error", "Box Filter error!!!")

def Gauss_Filter():#高斯濾波器函式
    try:
        cv.namedWindow('image')# 新建視窗
        cv.createTrackbar('kernel','image',3,20,nothing)
        cv.createTrackbar('kernel1','image',3,20,nothing)
        cv.createTrackbar('Deviation','image',1,20,nothing)
        while(1):
            r = cv.getTrackbarPos('kernel','image')
            r2 = cv.getTrackbarPos('kernel1','image')
            r1 = cv.getTrackbarPos('Deviation','image')
            if r%2==1 and r2%2==1:#設定如為偶數則不匯入
                gaussian = cv.GaussianBlur(im.im, (r, r2), r1)
                # 給予高斯化圖片的尺寸和標準差
            cv.imshow("image", gaussian)# 不斷更新當前圖片
            # 如按下ENTER將圖片匯入"影像處理程式開發平台"視窗
            k = cv.waitKey(1) & 0xFF
            if k == 13:
                cv.destroyWindow('image')
                return gaussian
            elif k==32:
                cv.destroyWindow('image')
                return im.im
    except Exception:
        msgbox.showerror("Error", "Gauss Filter error!!!")

def Median_Filter():#中值濾波器函式
    try:
        cv.namedWindow('image')# 新建視窗
        cv.createTrackbar('kernel','image',3,20,nothing)
        while(1):
            r = cv.getTrackbarPos('kernel','image')
            if r%2==1:# 設定如為偶數則不匯入
                median = cv.medianBlur(im.im, r)
                # 給予中值化圖片的尺寸和標準差
            cv.imshow("image", median)# 不斷更新當前圖片
            # 如按下ENTER將圖片匯入"影像處理程式開發平台"視窗
            k = cv.waitKey(1) & 0xFF
            if k == 13:
                cv.destroyWindow('image')
                return median
            elif k==32:
                cv.destroyWindow('image')
                return im.im
    except Exception:
        msgbox.showerror("Error", "Median Filter error!!!")

def Bilateral_Filter():#雙邊濾波器函數
    try:
        cv.namedWindow('image')# 新建視窗
        cv.createTrackbar('d','image',0,20,nothing)             #鄰域直徑
        cv.createTrackbar('sigmaColor','image',0,200,nothing)   #顏色標準差
        cv.createTrackbar('sigmaSpace','image',0,200,nothing)   #空間標準差
        while(1):
            d = cv.getTrackbarPos('d','image')                  #獲取數值
            sigmaColor = cv.getTrackbarPos('sigmaColor','image')
            sigmaSpace = cv.getTrackbarPos('sigmaSpace','image')
            dst = cv.bilateralFilter(im.im, d, sigmaColor, sigmaSpace)# 設定雙邊化圖片
            cv.imshow("image", dst)# 不斷更新當前圖片
            # 如按下ENTER將圖片匯入"影像處理程式開發平台"視窗
            k = cv.waitKey(1) & 0xFF
            if k == 13:
                cv.destroyWindow('image')
                return dst
            elif k==32:
                cv.destroyWindow('image')
                return im.im
    except Exception:
        msgbox.showerror("Error", "Median Filter error!!!")