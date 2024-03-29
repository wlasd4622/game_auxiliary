import time
import numpy as np
from PIL import ImageGrab
import cv2
import time


def screenshot():
    img = ImageGrab.grab()
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    nowStr = str(int(time.time()*1000))
    print(nowStr)
    cv2.imwrite("./temp/image2/"+nowStr+".png", img)



# 参数一:list结构的，轮廓信息
# 参数二:要使用的方法
# 返回值:处理过后的轮廓和矩形轮廓
def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0

    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    '''
    cv2.boundingRect(c)
    返回四个值，分别是x，y，w，h；
    x，y是矩阵左上点的坐标，w，h是矩阵的宽和高
    '''
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]  # 在轮廓信息中找到一个外接矩形
    (cnts, boundingBoxes) = zip(
        *sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    return cnts, boundingBoxes

# 重置大小，用于比较模板和图像中的数字是否一致
# 插值方法如下：
# INTER_NEAREST:最邻近插值
# INTER_LINEAR:双线性插值,默认情况下使用该方式进行插值.
# INTER_AREA:基于区域像素关系的一种重采样或者插值方式.该方法是图像抽取的首选方法,它可以产生更少的波纹,
# 但是当图像放大时,它的效果与INTER_NEAREST效果相似.
# INTER_CUBIC:4×4邻域双3次插值
# INTER_LANCZOS4:8×8邻域兰索斯插值

if __name__ == '__main__':
    while True:
        time.sleep(1)
        screenshot()
