import cv2 as cv
import numpy as np
from tkinter import messagebox as msgbox
from imutils import perspective
import im
def nothing(x):
    pass
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
        cv.destroyWindow('Warped')
        cv.destroyWindow('image')
        return warped
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
                cv.destroyWindow('image')
                return dst
            elif k==32:
                cv.destroyWindow('image')
                return im.im
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
                cv.destroyWindow('image')
                return dst
            elif k==32:
                cv.destroyWindow('image')
                return im.im  
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
                cv.destroyWindow('img')
                cv.destroyWindow('image')
                return dst
            elif k==32:
                cv.destroyWindow('img')
                cv.destroyWindow('image')
                return im.im  
    except Exception:
        msgbox.showerror("Error", "Affine Transform error!!!")