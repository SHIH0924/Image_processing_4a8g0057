import cv2 as cv
import numpy as np
from tkinter import messagebox as msgbox
from tkinter import filedialog
import im
def cv_imread(filePath):                    # 讀取檔案路徑
    cv_img=cv.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    return cv_img                           # 回傳路徑

def Harris_Corner_Detector():#哈里斯邊角偵測
    try:
        # 將輸入圖像轉換為灰度色彩空間
        operatedImage = cv.cvtColor(im.im, cv.COLOR_BGR2GRAY)
        #修改數據類型設置為 32 位浮點數
        operatedImage = np.float32(operatedImage)
        # 應用 cv2.cornerHarris 方法
        dest = cv.cornerHarris(operatedImage, 2, 5, 0.07)
        # 結果通過擴張的角標記
        dest = cv.dilate(dest, None)
        # 恢復到原始圖像
        im.im[dest > 0.01 * dest.max()]=[0, 0, 255]
        return im.im
    except Exception:
        msgbox.showerror("Error", "Median Filter error!!!")

def Canny_Edge_Detector():#邊緣偵測
    try:
        # 將輸入圖像轉換為灰度色彩空間
        gray = cv.cvtColor(im.im, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (5, 5), 0)
        # 應用 cv2.Canny 方法
        im.im = cv.Canny(blurred, 30, 150)
        return im.im
    except Exception:
        msgbox.showerror("Error", "Median Filter error!!!")

def Feature_Detector():#特徵偵測
    try:
        img = im.im
        sfname = filedialog.askopenfilename(title='選擇要開啟的檔案',
        # 開起對話框，對話框名稱
                                            filetypes=[# 文件能選擇的類型
                                                ('All Files','*'),
                                                ("jpeg files","*.jpg"),
                                                ("png files","*.png"),
                                                ("gif files","*.gif")])
        img2 = cv_imread(sfname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
        #特徵量
        detector = cv.ORB_create()
        kp1, des1 = detector.detectAndCompute(gray, None)
        kp2, des2 = detector.detectAndCompute(gray2, None)
        #比較器
        bf = cv.BFMatcher(cv.NORM_HAMMING)
        # 載入特徵點
        matches = bf.match(des1, des2)
        matches = sorted(matches, key = lambda x:x.distance)
        # 表示結果
        h1, w1, c1 = img.shape[:3]
        h2, w2, c2 = img2.shape[:3]
        height = max([h1,h2])
        width = w1 + w2
        out = np.zeros((height, width, 3), np.uint8)
        cv.drawMatches(img, kp1, img2, kp2, matches[:50],out, flags=0)
        cv.namedWindow('image')
        cv.imshow("image", out)
        while(1):
            k = cv.waitKey(1) & 0xFF
            if k == 13:
                cv.destroyWindow('image')
                return out
            elif k==32:
                cv.destroyWindow('image')
                return im.im
    except Exception:
        msgbox.showerror("Error", "Median Filter error!!!")

def SIFT_Feature_Description():#特徵描述
    try:
        x=im.im.copy()
        y=im.im.copy()
        img = x
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # 创建sift对象，调用detectAndCompute()函数
        sift = cv.SIFT_create()
        kp, des = sift.detectAndCompute(gray,None)
        # 画出特征点
        img = cv.drawKeypoints(img, kp, img)
        cv.imshow('image', img)
        img2 = y
        gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
        # 创建sift对象，调用detectAndCompute()函数
        sift2 = cv.SIFT_create()
        kp2, des2 = sift2.detectAndCompute(gray2,None)
        img2 = cv.drawKeypoints(img2, kp2, img2,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv.imshow('image2', img2)
        while(1):
            k = cv.waitKey(1) & 0xFF
            if k == 97:#按a匯入img
                cv.destroyWindow('image')
                cv.destroyWindow('image2')
                return img
            if k == 115:#按s匯入img2
                cv.destroyWindow('image')
                cv.destroyWindow('image2')
                return img2
            if k == 32:
                cv.destroyWindow('image')
                cv.destroyWindow('image2')
                return im.im
    except Exception:
        msgbox.showerror("Error", "Median Filter error!!!")
