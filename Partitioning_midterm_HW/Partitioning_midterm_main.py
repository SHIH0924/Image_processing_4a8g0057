import cv2 as cv
import tkinter as tk
from PIL import Image, ImageTk
from sympy import false, true                  
import im
import IO
import setting
import Image_Processing
import Geometric_transformation
import Neighborhood_processing
import Detector
import Contours
import Morphological_Transformations
#main
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
    im.im=intkimg# 將更改後的圖片放入im.im方便後續做其他更動

def open_file():                            # 開檔函式
    Intkinter(IO.open_file())#呼叫Intkinter函式將圖片匯入tkinter視窗
    
def save_file():                        # 存檔函式
    IO.save_file()

def change_color_space():#RGB轉HSV函數
    Intkinter(setting.change_color_space())

def RGB_To_Grayscale():#RGB轉灰階函式
    Intkinter(setting.RGB_To_Grayscale())

def Thresholding():# 影像二值化函式
    Intkinter(Image_Processing.Thresholding())

def opencv_histogram_equalizes():# 值方圖等化函式
    Intkinter(Image_Processing.opencv_histogram_equalizes())

# 透視投影轉換函式
def Perspective_Transform():
    Intkinter(Geometric_transformation.Perspective_Transform())

def Moving_Image():# 移動函式
    Intkinter(Geometric_transformation.Moving_Image())

def Rotate_The_Image():#旋轉函式
    Intkinter(Geometric_transformation.Rotate_The_Image())
        
def Affine_Transform():#仿射轉換函式
    Intkinter(Geometric_transformation.Affine_Transform())

def Mean_Filter():# 均值濾波器函式
    Intkinter(Neighborhood_processing.Mean_Filter())

def Box_Filter():# 方框濾波器函式
    Intkinter(Neighborhood_processing.Box_Filter())

def Gauss_Filter():#高斯濾波器函式
    Intkinter(Neighborhood_processing.Gauss_Filter())

def Median_Filter():#中值濾波器函式
    Intkinter(Neighborhood_processing.Median_Filter())

def Bilateral_Filter():#雙邊濾波器函數
    Intkinter(Neighborhood_processing.Bilateral_Filter())

def Harris_Corner_Detector():#哈里斯邊角偵測
    Intkinter(Detector.Harris_Corner_Detector())

def Canny_Edge_Detector():#邊緣偵測
    Intkinter(Detector.Canny_Edge_Detector())

def Feature_Detector():#特徵偵測
    Intkinter(Detector.Feature_Detector())

def SIFT_Feature_Description():#特徵描述
    Intkinter(Detector.SIFT_Feature_Description())

def Simple_Contour():#簡單輪廓
    Intkinter(Contours.Simple_Contour())

def Convex_Hull():#凸包
    Intkinter(Contours.Convex_Hull())

def Bounding_boxes():#邊界框
    Intkinter(Contours.Bounding_boxes())

def Dilating():#擴張
    Intkinter(Morphological_Transformations.Basic_morphology(cv.dilate))

def eroding():#侵蝕
    Intkinter(Morphological_Transformations.Basic_morphology(cv.erode))

def Skeletonize():#骨架化
    Intkinter(Morphological_Transformations.Skeletonize())

def perim():#邊緣
    Intkinter(Morphological_Transformations.perim())

def tophat():#執行形態學頂帽變換
    Intkinter(Morphological_Transformations.tophat())

def blackhat():#執行形態學黑帽變換
    Intkinter(Morphological_Transformations.blackhat())
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
list.add_command(label="儲存檔案", command=IO.save_file)# 子選單新增選項
menubar.add_cascade(label="File", menu=list)# 將子選單串接到父選單

list1=tk.Menu(menubar)
list1.add_command(label="設定ROI", command=setting.ROI)

list1_2=tk.Menu(menubar)# 在選單列下的子選單建立一個子選單
list1_2.add_command(label="影像大小", command=setting.Image_Size)
list1_2.add_command(label="顯示直方圖", command=setting.show_color_histogram)
list1.add_cascade(label="影像資訊呈現", menu=list1_2)

list1_1=tk.Menu(menubar)# 在選單列下的子選單建立一個子選單
list1_1.add_command(label="RGB轉HSV", command=change_color_space)
list1_1.add_command(label="RGB轉灰階", command=RGB_To_Grayscale)
list1.add_cascade(label="色彩空間轉換", menu=list1_1)

menubar.add_cascade(label="Setting", menu=list1)

list2=tk.Menu(menubar)                           
list2.add_command(label="影像二值化", command=Thresholding)
list2.add_command(label="直方圖等化", command=opencv_histogram_equalizes)


list2_1=tk.Menu(menubar)# 在選單列下的子選單建立一個子選單
list2_1.add_command(label="平移", command=Moving_Image)
list2_1.add_command(label="旋轉", command=Rotate_The_Image)
list2_1.add_command(label="仿射轉換", command=Affine_Transform)
list2_1.add_command(label="透視投影轉換", command=Perspective_Transform)
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
list3.add_command(label="Skeletonize", command=Skeletonize)
list3.add_command(label="perim", command=perim)
list3.add_command(label="tophat", command=tophat)
list3.add_command(label="blackhat", command=blackhat)
menubar.add_cascade(label="Detector", menu=list3)

menubar.add_command(label="Quit", command=win.destroy)
win.config(menu=menubar)# 設定視窗的選單列
win.mainloop()# 重覆執行全程式並不斷重覆