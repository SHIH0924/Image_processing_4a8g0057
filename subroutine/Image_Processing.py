import cv2 as cv
from tkinter import messagebox as msgbox
import im
def nothing(x):
    pass
def Thresholding():# 影像二值化函式
    try:
        cv.namedWindow('image')
        cv.createTrackbar('value','image',-10,255,nothing)
        cv.createTrackbar('Brightness','image',0,255,nothing)
        while(1):
            r = cv.getTrackbarPos('value','image')
            g = cv.getTrackbarPos('Brightness','image')
            ret,thresh=cv.threshold(im.im,r,g,cv.THRESH_BINARY)
            # 將小於閾值的灰度值設為0，其他值設為最大灰度值。
            cv.imshow('image',thresh)
            # 顯示當前滾動條的數值顯示出的二值畫圖片
            k = cv.waitKey(1) & 0xFF
            if k == 13:
                cv.destroyWindow('image')
                return thresh
            elif k==32:
                cv.destroyWindow('image')
                return im.im
    except Exception:
        msgbox.showerror("Error", "Thresholding error!!!")

def opencv_histogram_equalizes():# 值方圖等化函式
    try:
        ycrcb = cv.cvtColor(im.im, cv.COLOR_BGR2YCR_CB)# 轉換為YCrCb圖像
        channels = cv.split(ycrcb)# 分裂出三個單通道圖像分別為的B、G、R
        cv.equalizeHist(channels[0], channels[0])# 將圖片均衡化
        cv.merge(channels, ycrcb)# 合成channels, ycrcb
        image=cv.cvtColor(ycrcb, cv.COLOR_YCR_CB2BGR, im.im)
        # 將圖片匯入"影像處理程式開發平台"視窗
        return image
    except Exception:
        msgbox.showerror("Error", "Histogram equalized error!!!")