import cv2 as cv
import numpy as np
from tkinter import messagebox as msgbox
from tkinter import filedialog
from matplotlib import pyplot as plt
import im
def ROI():                                  # ROI函式
    try:
        g=im.im                             # 讀入當前圖片
        showCrosshair = True                # 是否顯示網格
        fromCenter = False# 拉選區域時滑鼠從左上角到右下角選取區域
        rect = cv.selectROI("crop window", g, showCrosshair, fromCenter)
        # 選擇ROI，按下ENTER確定選取
        print("select area")                # 顯示select area
        (x, y, w, h) = rect                 # 記錄數值及位置
        imCrop = g[y : y+h, x:x+w]          # 裁剪圖像
        cv.imshow("image_roi", imCrop)      # 顯示裁剪後的圖像
        while(1):                           # 存檔
            k = cv.waitKey(1) & 0xFF
            if k == 13:# 如果按下ENTER則跳出存檔對話框
                fname = filedialog.asksaveasfilename(
                                                title=u'保存文件', 
                                                filetypes=[
                                                ('All Files','*'),
                                                ("jpeg files","*.jpg"),
                                                ("png files","*.png"),
                                                ("gif files","*.gif")])
                cv.imwrite(fname+'.jpg', imCrop)
                cv.destroyWindow('image_roi')   # 存檔後關閉裁剪後的圖像
                cv.destroyWindow('crop window') # 存檔後關閉裁剪視窗
                break                           # 跳出迴圈
            elif k==32:
                cv.destroyWindow('image_roi')   # 關閉裁剪後的圖像
                cv.destroyWindow('crop window') # 關閉裁剪視窗
                break                           # 跳出迴圈
    except Exception:
        msgbox.showerror("Error", "File ROI error!!!")

def Image_Size():#影像大小函式
    try:
        size = im.im.shape#取得影像資訊
        mess="長:%d\n寬:%d\n色彩通道數:%d"%(size[0],size[1],size[2])
        #將取得的資訊放入字串中
        msgbox.showinfo("Image size", mess)# 跳出對話框
    except Exception:
        msgbox.showerror("Error", "Not Image !!!")

def show_color_histogram():                     # 劃出彩色直方圖
    try:
        color=('b','g','r')                     # 分為三個顏色
        for i,col in enumerate(color):          # 依序製作三個顏色的線
            hist=cv.calcHist([im.im],[i],None,[256],[0,256])# 設定直方圖
            plt.plot(hist,color=col)            # (一維陣列,線顏色)
            plt.xlim([0,256])                   # x範圍的值
        plt.show()                              # 顯示出圖表
    except Exception:
        msgbox.showerror("Error", "Show color histogram error!!!")
        
def nothing(x):
    pass
def change_color_space():#RGB轉HSV函數
    try:
        cv.namedWindow('image')                 # 新建視窗
        cv.createTrackbar('H_h','image',179,179,nothing)# 創建滾動條
        cv.createTrackbar('S_h','image',255,255,nothing)
        cv.createTrackbar('V_h','image',255,255,nothing)
        cv.createTrackbar('H_l','image',0,179,nothing)
        cv.createTrackbar('S_l','image',0,255,nothing)
        cv.createTrackbar('V_l','image',0,255,nothing)
        while(1):# 獲取滾動條的數值
            r_number_h = cv.getTrackbarPos('H_h','image')
            g_number_h = cv.getTrackbarPos('S_h','image')
            b_number_h = cv.getTrackbarPos('V_h','image')
            r_number_l = cv.getTrackbarPos('H_l','image')
            g_number_l = cv.getTrackbarPos('S_l','image')
            b_number_l = cv.getTrackbarPos('V_l','image')
            lower = np.array([r_number_l, g_number_l, b_number_l])
            # 設置過濾的顏色低值
            upper = np.array([r_number_h, g_number_h, b_number_h])
            # 設置過濾的顏色高值
            hsv = cv.cvtColor(im.im, cv.COLOR_BGR2HSV)# 將圖片轉成 hsv
            mask = cv.inRange(hsv, lower, upper)
            # 調節圖片颜色信息、飽和度、亮度區間
            out=cv.bitwise_and(im.im,im.im,mask=mask)# 做and操作
            cv.imshow('image',out)# 顯示出圖片
            # 如果按下ENTER則將圖片匯入"影像處理程式開發平台"視窗
            k = cv.waitKey(1) & 0xFF
            if k == 13:
                cv.destroyWindow('image')
                return out
            elif k==32:
                cv.destroyWindow('image')
                return im.im   
    except Exception:
        msgbox.showerror("Error", "Change color space error!!!")

def RGB_To_Grayscale():#RGB轉灰階函式
    try:
        image = cv.cvtColor(im.im, cv.COLOR_BGR2GRAY)#將RGB轉灰色
        # 將圖片匯入"影像處理程式開發平台"視窗
        return image
    except Exception:
        msgbox.showerror("Error", "RGB to grayscale error!!!")