import cv2 as cv
import numpy as np
from tkinter import messagebox as msgbox
from tkinter import filedialog
import im
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
        return im.im
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