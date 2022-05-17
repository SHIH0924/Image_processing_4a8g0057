import cv2 as cv
from cv2 import IMREAD_GRAYSCALE
import numpy as np
import tkinter as tk
from tkinter import messagebox as msgbox
from tkinter import filedialog
from PIL import Image, ImageTk
from matplotlib import pyplot as plt
from sympy import false, true                  
import im
from imutils import perspective

def Intkinter(intkimg):
    cv2image = cv.cvtColor(intkimg, cv.COLOR_BGR2RGBA)
    # 將顏色由 BRG 轉為 RGB
    img = Image.fromarray(cv2image)
    # OpenCV轉換PIL.Image格式
    imgtk = ImageTk.PhotoImage(image=img)
    # 轉換成Tkinter可以用的圖片
    video.imgtk = imgtk
    # 將imgtk丟入video.imgtk
    video.configure(image=imgtk)
    # 放入圖片
    im.im=intkimg
    # 將更改後的圖片放入im.im方便後續做其他更動

def cv_imread(filePath):                    # 讀取檔案路徑
    cv_img=cv.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    return cv_img                           # 回傳路徑

def open_file():                            # 開檔函式
    try:                                    # 如果沒有錯誤則執行
        sfname = filedialog.askopenfilename(title='選擇要開啟的檔案',
        # 開起對話框，對話框名稱
                                            filetypes=[# 文件能選擇的類型
                                                ('All Files','*'),
                                                ("jpeg files","*.jpg"),
                                                ("png files","*.png"),
                                                ("gif files","*.gif")])
        im.im = cv_imread(sfname)
        # 讀取路徑後丟入im.im
        Intkinter(im.im)#呼叫Intkinter函式將圖片匯入tkinter視窗
    except Exception:# 如果有錯誤則執行
        msgbox.showerror("Error", "File opening error!!!")
        # 跳出對話框
    
def save_file():                        # 存檔函式
    try:                                # 如果沒有錯誤則執行
        fname =filedialog.asksaveasfilename(title=u'保存文件',
        # 開起對話框，對話框名稱
                                            filetypes=[# 文件能選擇的類型
                                                ('All Files','*'),
                                                ("jpeg files","*.jpg"),
                                                ("png files","*.png"),
                                                ("gif files","*.gif")])
        cv.imwrite(fname+'.jpg' , im.im)# 要儲存的檔名及檔案
    except Exception:
        msgbox.showerror("Error", "File save error!!!")

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
                cv.imwrite("image_roi.png", imCrop)
                fname = tk.filedialog.asksaveasfilename(
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
                Intkinter(out)
                cv.destroyWindow('image')
                break    
    except Exception:
        msgbox.showerror("Error", "Change color space error!!!")

def RGB_To_Grayscale():#RGB轉灰階函式
    try:
        image = cv.cvtColor(im.im, cv.COLOR_BGR2GRAY)#將RGB轉灰色
        # 將圖片匯入"影像處理程式開發平台"視窗
        Intkinter(image)
    except Exception:
        msgbox.showerror("Error", "RGB to grayscale error!!!")

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
                # 如果按下ENTER則將圖片匯入"影像處理程式開發平台"視窗
                Intkinter(thresh)
                cv.destroyWindow('image')
                break
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
        Intkinter(image)
    except Exception:
        msgbox.showerror("Error", "Histogram equalized error!!!")

#定義點擊事件
def on_EVENT_LBUTTONDOWN(event, x, y,flags, param):
    try:
        if event == cv.EVENT_LBUTTONDOWN:# 如果存在滑鼠點擊事件
            xy = "%d,%d" % (x, y)        # 得到坐標x,y
            im.a.append(x)     # 將每次的坐標存放在a陣列里面
            im.b.append(y)     # 將每次的坐標存放在b陣列里面
            cv.circle(xx, (x, y), 10, (0, 0, 255), thickness=-1)  
            # 點擊的地方小紅圓點顯示
            cv.putText(xx, xy, (x, y), cv.FONT_HERSHEY_PLAIN,
                                1.0, (0, 0, 0), thickness=1)
            # 點擊的地方顯示坐標數字 引數1圖片，引數2添加的文字
            # 引數3左上角坐標，引數4字體，引數5字體粗細
            cv.imshow("image", xx)    #顯示圖片
    except Exception:
        msgbox.showerror("Error", "on_EVENT_LBUTTONDOWN error!!!")

# 透視投影轉換函式
def Perspective_Transform():
    try:
        im.a = []                                  # 用於存放橫坐標
        im.b = []                                  # 用於存放縱坐標
        cv.namedWindow("image")             # 定義圖片視窗
        cv.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
        # 回呼函式，引數1視窗的名字，引數2滑鼠回應函式
        global xx                           # 用於存放要更改的圖片
        xx = im.im.copy()
        cv.imshow("image", xx)              # 顯示圖片
        cv.waitKey(0)                       # 不斷重繪影像
        c = []                              # 用於存放所有坐標
        for i in range(0,len(im.a)):
            print(im.a[i], im.b[i])               # 列印坐標
            c.append([im.a[i], im.b[i]])          # 加入c中
        print(c)                            # 印出選擇的點
        clone = im.im.copy()
        pts = np.array(c)
        warped = perspective.four_point_transform(clone, pts)
        # 應用四點變換獲得“鳥瞰圖”
        cv.imshow("Warped", warped)# 顯示透視投影後影像
        cv.waitKey(0)#按下ENTER繼續向下執行
        Intkinter(warped)
        cv.destroyWindow('Warped')
        cv.destroyWindow('image')
    except Exception:
        msgbox.showerror("Error", "Perspective Transform error!!!")

def Moving_Image():# 移動函式
    try:
        rows,cols = im.im.shape[:2] # 獲取圖片長寬
        cv.namedWindow('image')     # 新建視窗
        cv.createTrackbar('about','image',cols,cols*2,nothing)
        cv.createTrackbar('seesaw','image',rows,rows*2,nothing)
        while(1):
            s = cv.getTrackbarPos('about','image')
            a = cv.getTrackbarPos('seesaw','image')
            M = np.float32([[1,0,s-cols],[0,1,a-rows]])# 定義平移矩陣M
            dst = cv.warpAffine(im.im,M,(cols,rows))#將產生的矩陣M賦值給仿射函數
            cv.imshow("image", dst)#不斷更新當前圖片
            # 如按下ENTER將圖片匯入"影像處理程式開發平台"視窗
            k = cv.waitKey(1) & 0xFF
            if k == 13:
                Intkinter(dst)
                cv.destroyWindow('image')
                break
    except Exception:
        msgbox.showerror("Error", "Moving Image error!!!")

def Rotate_The_Image():#旋轉函式
    try:
        rows,cols = im.im.shape[:2] # 獲取圖片長寬
        cv.namedWindow('image')     # 新建視窗
        cv.createTrackbar('angle','image',0,360,nothing)
        cv.createTrackbar('size(/10)','image',10,20,nothing)
        while(1):
            a = cv.getTrackbarPos('angle','image')
            s = cv.getTrackbarPos('size(/10)','image')
            M = cv.getRotationMatrix2D((cols/2,rows/2),a,s/10)# 定義旋轉矩陣M
            dst = cv.warpAffine(im.im,M,(cols,rows))#將產生的矩陣M賦值給仿射函數
            cv.imshow("image", dst)#不斷更新當前圖片
            # 如按下ENTER將圖片匯入"影像處理程式開發平台"視窗
            k = cv.waitKey(1) & 0xFF
            if k == 13:
                Intkinter(dst)
                cv.destroyWindow('image')
                break
    except Exception:
        msgbox.showerror("Error", "Rotate The Image error!!!")
def Affine_Transform():#仿射轉換函式
    try:
        height, width = im.im.shape[:2]# 獲取圖片長寬
        cv.namedWindow('image',0) # 新建視窗
        cv.resizeWindow("image", 400, 770)#設定視窗大小
        cv.createTrackbar('originalX1','image',0,width,nothing)
        cv.createTrackbar('originalY1','image',0,height,nothing)
        cv.createTrackbar('originalX2','image',width,width,nothing)
        cv.createTrackbar('originalY2','image',0,height,nothing)
        cv.createTrackbar('originalX3','image',0,width,nothing)
        cv.createTrackbar('originalY3','image',height,height,nothing)
        cv.createTrackbar('targetX1','image',0,width,nothing)
        cv.createTrackbar('targetY1','image',0,height,nothing)
        cv.createTrackbar('targetX2','image',width,width,nothing)
        cv.createTrackbar('targetY2','image',0,height,nothing)
        cv.createTrackbar('targetX3','image',0,width,nothing)
        cv.createTrackbar('targetY3','image',height,height,nothing)
        while(1):
            ox1 = cv.getTrackbarPos('originalX1','image')
            oy1 = cv.getTrackbarPos('originalY1','image')
            ox2 = cv.getTrackbarPos('originalX2','image')
            oy2 = cv.getTrackbarPos('originalY2','image')
            ox3 = cv.getTrackbarPos('originalX3','image')
            oy3 = cv.getTrackbarPos('originalY3','image')

            tx1 = cv.getTrackbarPos('targetX1','image')
            ty1 = cv.getTrackbarPos('targetY1','image')
            tx2 = cv.getTrackbarPos('targetX2','image')
            ty2 = cv.getTrackbarPos('targetY2','image')
            tx3 = cv.getTrackbarPos('targetX3','image')
            ty3 = cv.getTrackbarPos('targetY3','image')

            # 在原圖像和目標圖像上各選擇三個點 
            mat_src = np.float32([[ox1,oy1],[ox2,oy2],[ox3,oy3]]) 
            mat_dst = np.float32([[tx1,ty1],[tx2,ty2],[tx3,ty3]]) 
            mat_trans = cv.getAffineTransform(mat_src, mat_dst)# 得到變換矩陣
            dst = cv.warpAffine(im.im, mat_trans, (width,height))# 進行仿射變換 
            cv.imshow("img", dst)# 不斷更新當前圖片
            # 如按下ENTER將圖片匯入"影像處理程式開發平台"視窗
            k = cv.waitKey(1) & 0xFF
            if k == 13:
                Intkinter(dst)
                cv.destroyWindow('img')
                cv.destroyWindow('image')
                break
    except Exception:
        msgbox.showerror("Error", "Affine Transform error!!!")
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
                Intkinter(blur)
                cv.destroyWindow('image')
                break
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
                Intkinter(box)
                cv.destroyWindow('image')
                break
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
                Intkinter(gaussian)
                cv.destroyWindow('image')
                break
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
                Intkinter(median)
                cv.destroyWindow('image')
                break
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
                Intkinter(dst)
                cv.destroyWindow('image')
                break
    except Exception:
        msgbox.showerror("Error", "Median Filter error!!!")

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
        Intkinter(im.im)
    except Exception:
        msgbox.showerror("Error", "Median Filter error!!!")

def Canny_Edge_Detector():#邊緣偵測
    try:
        # 將輸入圖像轉換為灰度色彩空間
        gray = cv.cvtColor(im.im, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (5, 5), 0)
        # 應用 cv2.Canny 方法
        im.im = cv.Canny(blurred, 30, 150)
        Intkinter(im.im)
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
                Intkinter(out)
                cv.destroyWindow('image')
                break
            elif k==32:
                cv.destroyWindow('image')
                break
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
            if k == 97:
                Intkinter(img)
                cv.destroyWindow('image')
                cv.destroyWindow('image2')
                break
            if k == 115:
                Intkinter(img2)
                cv.destroyWindow('image')
                cv.destroyWindow('image2')
                break
            if k == 32:
                cv.destroyWindow('image')
                cv.destroyWindow('image2')
                break
    except Exception:
        msgbox.showerror("Error", "Median Filter error!!!")

def Simple_Contour():#簡單輪廓
    try:
        image = im.im
        # 將圖像轉換為灰階格式
        img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # 應用二進制閾值
        ret, thresh = cv.threshold(img_gray, 150, 255, cv.THRESH_BINARY)
        # 使用 cv2.CHAIN_APPROX_NONE 檢測二值圖像上的輪廓
        contours, hierarchy = cv.findContours(image=thresh, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
        # 在原始圖像上繪製輪廓
        cv.namedWindow('image')# 新建視窗
        cv.createTrackbar('B','image',0,255,nothing)             #鄰域直徑
        cv.createTrackbar('G','image',0,255,nothing)   #顏色標準差
        cv.createTrackbar('R','image',0,255,nothing)   #空間標準差
        cv.createTrackbar('L','image',0,20,nothing)   #空間標準差
        while(1):
            B = cv.getTrackbarPos('B','image')                  #獲取數值
            G = cv.getTrackbarPos('G','image')
            R = cv.getTrackbarPos('R','image')
            L = cv.getTrackbarPos('L','image')
            image_copy = image.copy()
            cv.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(B, G, R), thickness=L, lineType=cv.LINE_AA)
            cv.imshow('image',image_copy)
            k = cv.waitKey(1) & 0xFF
            if k == 13:
                Intkinter(image_copy)
                cv.destroyWindow('image')
                break
            elif k==32:
                cv.destroyWindow('image')
                break
    except Exception:
        msgbox.showerror("Error", "Median Filter error!!!")

def Convex_Hull():#凸包
    try:
        image = im.im
        # 將圖像轉換為灰度格式
        img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # 應用二進制閾值
        ret, thresh = cv.threshold(img_gray, 150, 255, cv.THRESH_BINARY)
        # 使用 cv2.CHAIN_APPROX_NONE 檢測二值圖像上的輪廓
        contours, hierarchy = cv.findContours(image=thresh, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
        hull = []
        # 計算每個輪廓的點
        for i in range(len(contours)):
            # creating convex hull object for each contour
            hull.append(cv.convexHull(contours[i], False))
        # 創建一個空的黑色圖像
        drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)
        # 繪製輪廓和船體點
        for i in range(len(contours)):
            color_contours = (0, 255, 0) # green - 輪廓顏色
            color = (255, 0, 0) # blue - 凸包顏色
            # draw ith contour
            cv.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
            # 繪製凸包對象
            cv.drawContours(drawing, hull, i, color, 1, 8)
        # 查看結果
        cv.imshow('Convex Hull', drawing)
        while(1):
            k = cv.waitKey(1) & 0xFF
            if k == 13:
                Intkinter(drawing)
                cv.destroyWindow('Convex Hull')
                break
            elif k==32:
                cv.destroyWindow('Convex Hull')
                break
    except Exception:
        msgbox.showerror("Error", "Median Filter error!!!")

def Bounding_boxes():#邊界框
    try:
        img = im.im.copy()
        ret, threshed_img = cv.threshold(cv.cvtColor(img, cv.COLOR_BGR2GRAY),
                        127, 255, cv.THRESH_BINARY)
        # 找到輪廓並獲取外部輪廓
        contours, hier = cv.findContours(threshed_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # 對於每個輪廓，用綠色繪製 boundingRect
        # 一個紅色的 minAreaRect 和
        # 一個藍色的 minEnclosureCircle
        for c in contours:
            # 獲取邊界矩形
            x, y, w, h = cv.boundingRect(c)
            # 繪製一個綠色矩形來可視化邊界矩形
            cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # 得到最小面積矩形
            rect = cv.minAreaRect(c)
            box = cv.boxPoints(rect)
            # 將所有坐標浮點值轉換為 int
            box = np.int0(box)
            # 畫一個紅色的“nghien”矩形
            cv.drawContours(img, [box], 0, (0, 0, 255))

            # 最後，得到最小的封閉圓
            (x, y), radius = cv.minEnclosingCircle(c)
            # convert all values to int
            center = (int(x), int(y))
            radius = int(radius)
            # 並用藍色繪製圓圈
            img = cv.circle(img, center, radius, (255, 0, 0), 2)

        cv.drawContours(img, contours, -1, (255, 255, 0), 1)
        cv.imshow("Bounding boxes", img)
        while(1):
            k = cv.waitKey(1) & 0xFF
            if k == 13:
                Intkinter(img)
                cv.destroyWindow('Bounding boxes')
                break
            elif k==32:
                cv.destroyWindow('Bounding boxes')
                break
    except Exception:
        msgbox.showerror("Error", "Median Filter error!!!")

def Basic_morphology(x):
    cv.x=x
    try:
        cv.namedWindow('image')# 新建視窗
        cv.createTrackbar('frequency','image',0,20,nothing)
        cv.createTrackbar('x','image',0,20,nothing)
        cv.createTrackbar('y','image',0,20,nothing)
        while(1):
            frequency = cv.getTrackbarPos('frequency','image')
            x = cv.getTrackbarPos('x','image')
            y = cv.getTrackbarPos('y','image')
            img=im.im
            kernal=np.ones((x,y),np.uint8)
            des=cv.x(img,kernal,iterations=frequency)
            cv.imshow('image',des)
            k = cv.waitKey(1) & 0xFF
            if k == 13:
                Intkinter(des)
                cv.destroyWindow('image')
                break
            elif k==32:
                cv.destroyWindow('image')
                break
    except Exception:
        msgbox.showerror("Error", "Median Filter error!!!")
def Dilating():#擴張
    Basic_morphology(cv.dilate)

def eroding():#侵蝕
    Basic_morphology(cv.erode)

def perim():#邊緣
    try:
        input_image = im.im
        img = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY) 
        cv.namedWindow('image')# 新建視窗
        cv.createTrackbar('x','image',10,200,nothing)
        cv.createTrackbar('y','image',10,200,nothing)
        while(1):
            x = cv.getTrackbarPos('x','image')
            y = cv.getTrackbarPos('y','image')
            kernel = np.ones((x,y),np.uint8)
            dilation = cv.dilate(img,kernel,iterations = 1)
            diff2 = dilation - img
            cv.imshow("image", diff2)
            k = cv.waitKey(1) & 0xFF
            if k == 13:
                Intkinter(diff2)
                cv.destroyWindow('image')
                break
            elif k==32:
                cv.destroyWindow('image')
                break
    except Exception:
        msgbox.showerror("Error", "Median Filter error!!!")

def tophat():#執行形態學頂帽變換
    try:
        input_image = im.im
        input_image = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY) 
        cv.namedWindow('image')# 新建視窗
        cv.createTrackbar('x','image',10,200,nothing)
        cv.createTrackbar('y','image',10,200,nothing)
        while(1):
            x = cv.getTrackbarPos('x','image')
            y = cv.getTrackbarPos('y','image')
            # 獲取要在 Top-Hat 中使用的內核
            filterSize =(x, y)
            kernel = cv.getStructuringElement(cv.MORPH_RECT, filterSize)
            # 應用Top-Hat操作
            tophat_img = cv.morphologyEx(input_image, cv.MORPH_TOPHAT,kernel)
            cv.imshow("image", tophat_img)
            k = cv.waitKey(1) & 0xFF
            if k == 13:
                Intkinter(tophat_img)
                cv.destroyWindow('image')
                break
            elif k==32:
                cv.destroyWindow('image')
                break
    except Exception:
        msgbox.showerror("Error", "Median Filter error!!!")

win=tk.Tk()                             # 宣告一視窗
win.title("影像處理程式開發平台")        # 視窗名稱
win.geometry("750x500")                 # 視窗大小(寬x高)
win.resizable(true, true)               # 設定視窗能否調整大小
videoFrame = tk.Frame(win).pack()       # 切分視窗以方便進行排版
video = tk.Label(videoFrame)            # 將切分好的視窗建成一個標籤
video.pack()                            # 將上面兩行顯示出來

menubar=tk.Menu(win)                    # 建立頂層父選單 (選單列)
list=tk.Menu(menubar)                   # 在選單列下建立一個子選單
list.add_command(label="開啟檔案", command=open_file)# 子選單新增選項
list.add_command(label="儲存檔案", command=save_file)# 子選單新增選項
menubar.add_cascade(label="File", menu=list)# 將子選單串接到父選單

list1=tk.Menu(menubar)
list1.add_command(label="設定ROI", command=ROI)

list1_2=tk.Menu(menubar)# 在選單列下的子選單建立一個子選單
list1_2.add_command(label="影像大小", command=Image_Size)
list1_2.add_command(label="顯示直方圖", command=show_color_histogram)
list1.add_cascade(label="影像資訊呈現", menu=list1_2)

list1_1=tk.Menu(menubar)# 在選單列下的子選單建立一個子選單
list1_1.add_command(label="RGB轉HSV", command=change_color_space)
list1_1.add_command(label="RGB轉灰階", command=RGB_To_Grayscale)
list1.add_cascade(label="色彩空間轉換", menu=list1_1)

menubar.add_cascade(label="Setting", menu=list1)

list2=tk.Menu(menubar)                           
list2.add_command(label="影像二值化", command=Thresholding)
list2.add_command(label="直方圖等化", command=opencv_histogram_equalizes)
list2.add_command(label="透視投影轉換", command=Perspective_Transform)

list2_1=tk.Menu(menubar)# 在選單列下的子選單建立一個子選單
list2_1.add_command(label="平移", command=Moving_Image)
list2_1.add_command(label="旋轉", command=Rotate_The_Image)
list2_1.add_command(label="仿射轉換", command=Affine_Transform)
list2.add_cascade(label="幾何轉換功能", menu=list2_1)

list2_2=tk.Menu(menubar)# 在選單列下的子選單建立一個子選單
list2_2.add_command(label="均值濾波器", command=Mean_Filter)
list2_2.add_command(label="方框濾波器", command=Box_Filter)
list2_2.add_command(label="高斯濾波器", command=Gauss_Filter)
list2_2.add_command(label="中值濾波器", command=Median_Filter)
list2_2.add_command(label="雙邊濾波器", command=Bilateral_Filter)
list2.add_cascade(label="鄰域處理功能", menu=list2_2)
menubar.add_cascade(label="Image Processing", menu=list2)

list3=tk.Menu(menubar)                           
list3.add_command(label="Harris Corner Detector", command=Harris_Corner_Detector)
list3.add_command(label="Canny Edge Detector", command=Canny_Edge_Detector)
list3.add_command(label="Feature Detector", command=Feature_Detector)
list3.add_command(label="SIFT Feature Description", command=SIFT_Feature_Description)
list3.add_separator()
list3.add_command(label="Simple Contour", command=Simple_Contour)
list3.add_command(label="Convex Hull", command=Convex_Hull)
list3.add_command(label="Bounding boxes", command=Bounding_boxes)
list3.add_separator()
list3.add_command(label="Dilating", command=Dilating)
list3.add_command(label="eroding", command=eroding)
list3.add_command(label="perim", command=perim)
list3.add_command(label="tophat", command=tophat)
menubar.add_cascade(label="Detector", menu=list3)

menubar.add_command(label="Quit", command=win.destroy)
win.config(menu=menubar)# 設定視窗的選單列
win.mainloop()# 重覆執行全程式並不斷重覆