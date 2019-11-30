import time
import numpy as np
from PIL import ImageGrab
import cv2
import time
import utils
import km


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imshow('erosion/dilation', np.hstack((img, erosion, dilation)))


def imgHandle(img, feature):
    resultArr = []
    # 灰度
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 二值
    ret, th = cv2.threshold(
        grayImg, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 膨胀
    kernel = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(th.copy(), kernel)  # 膨胀
    # cv_show('dilation', dilation)
    # 轮廓
    refCnts, hierarchy = cv2.findContours(
        dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    refCnts = utils.sort_contours(
        refCnts, method="left-to-right")[0]  # 排序从左到右，从上到下

    for(i, c) in enumerate(refCnts):
        # 计算外接矩形并且resize成合适大小
        (x, y, w, h) = cv2.boundingRect(c)
        if w > 20 and w < 26 and h > 18 and h < 26:
            cv2.rectangle(th, (x, y), (x + w, y + h), (0, 0, 0), 1)
            # cv_show('img', th[y:y+h, x:x+w])
            resultArr.append(discriminate(th[y:y+h, x:x+w], feature))
    # cv_show('img', th)
    # 分割

    # 识别方向

    # 返回结果

    return resultArr


def discriminate(img, feature):
    scores = []
    for f in feature:
        result = cv2.matchTemplate(img, f, cv2.TM_CCOEFF)
        (_, score, _, _) = cv2.minMaxLoc(result)
        scores.append(score)

    return np.argmax(scores)


def init():
    feature = []
    upperImg = cv2.imread('./feature/upper.png')
    rightImg = cv2.imread('./feature/right.png')
    downImg = cv2.imread('./feature/down.png')
    leftImg = cv2.imread('./feature/left.png')
    arr = [upperImg, rightImg, downImg, leftImg]
    for img in arr:
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 二值
        ret, th = cv2.threshold(
            grayImg, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        feature.append(th)
        # cv_show('th', th)
    return feature


def main(feature):
    img = cv2.imread('./temp/image2/1575109194327.png')
    x = 540
    y = 420
    img = img[y:y+100, x:x+310]
    # cv_show('img', img)
    result = imgHandle(img, feature)
    print(result)


if __name__ == '__main__':
    # feature = init()
    # main(feature)
    km.MoveTo(500, 500)
    km.LeftClick()
    km.SayString("644615565")
