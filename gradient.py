import cv2 as cv
import copy
import numpy as np


def gaussian():
    img = cv.imread('elev2.jpeg')
    # 高斯滤波
    # img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img = np.float32(img) / 255.0  # 归一化
    img_Guassian = cv.GaussianBlur(img, (5, 5), 0)
    return img_Guassian


def conv1D(img, mask1D):
    width = img.shape[1]
    height = img.shape[0]
    # ximg，yimg表示x，y方向梯度图
    # 此处的变量赋值不能简单的等于，那样只会全部指向img
    ximg = copy.deepcopy(img)
    yimg = copy.deepcopy(img)
    # 计算x，y方向梯度
    for i in range(1, height - 2):
        for j in range(1, width - 2):
            left = img[i][j - 1]
            right = img[i][j + 1]
            uper = img[i + 1][j]
            lower = img[i - 1][j]
            # 如果用灰度图的话，此处int(right)应该加上,彩色图片不用加
            ximg[i][j] = right - left
            yimg[i][j] = lower - uper

    return ximg, yimg


def convertXYGradToVec(ximg, yimg):
    # 这个是最大范数的颜色角标
    maxIndex = 0
    width = ximg.shape[1]
    height = ximg.shape[0]
    vecImg = np.zeros((height,width,2))
    grayImg = np.zeros((height, width))
    for i in range(1, height - 2):
        for j in range(1, width - 2):
            bnorm = np.linalg.norm([ximg[i][j][0], yimg[i][j][0]])
            gnorm = np.linalg.norm([ximg[i][j][1], yimg[i][j][1]])
            rnorm = np.linalg.norm([ximg[i][j][2], yimg[i][j][2]])
            maxIndex = findMaxIn3(bnorm, gnorm, rnorm)
            vecImg[i][j]=[ximg[i][j][maxIndex], yimg[i][j][maxIndex]]
            grayImg[i][j] = float(np.linalg.norm([ximg[i][j][maxIndex], yimg[i][j][maxIndex]]))

    return grayImg


# 求三通道那个最大？返回通道下标b,g,r
def findMaxIn3(a, b, c):
    max = -255
    index = 0
    if a > max:
        max = a
        index = 0
    if b > max:
        max = b
        index = 1
    if c > max:
        max = c
        index = 2
    return index


if __name__ == "__main__":
    img_guassian = gaussian()
    # 求梯度也可以用以下sobel
    # gx = cv.Sobel(img_guassian, cv.CV_32F, 1, 0, ksize=1)
    ximg_conv, yimg_conv = conv1D(img_guassian, [-1, 0, 1])
    vec_img=convertXYGradToVec(ximg_conv, yimg_conv)
    cv.imshow("/mine", vec_img)
    # cv.imshow("/their", gx)
    cv.waitKey(0)
    cv.destroyAllWindows()
