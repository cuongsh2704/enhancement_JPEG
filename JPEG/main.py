import cv2
import numpy as np
def ShowImg(img1, img2):
    cv2.imshow("title", img1)
    cv2.imshow("title2", img2)
    cv2.waitKey(0)
def filterD(img, x, y, size, r): #hodges_lehman D,size : size of window , r : range , img(x,y)
    space = int(size/2)
    center = int(img[x][y][1])
    D = []
    for u in range(x - space, x + space + 1):
        for v in range(y - space, y + space + 1):
            if (img[u][v][1] <= center + r and img[u][v][1] >= center - r ):
                D.append(img[u][v][1])

    D = sorted(D)
    Ave = []
    N = int(len(D) / 2)
    lenD = len(D)
    for i in range(0, N):
        value = (int(D[i]) + int(D[lenD - i -2])) / 2
        Ave.append(value)

    Med = Median(Ave)

    img[x][y][0] = Med
    img[x][y][1] = Med
    img[x][y][2] = Med
    return img
def Median(arr):
    sumar = sum(arr)
    l = len(arr)
    if (l == 0) :
        l = 1
    return int(sumar / l)
def flatImg(img):
    (h, w, d) = img.shape
    img_flat = np.zeros((h, w), dtype= int)
    for i in range(0, h - 1):
        for j in range(0, w - 1):
            img_flat[i][j] = img[i][j][1]
    return img_flat
def  MSM_filter(img, x, y, size):  #multi_stage_Median filter
    space = int(size / 2)
    W1 = 0
    W2 = 0
    W3 = 0
    W4 = 0
    for i in range(-space, space + 1):
        W1 += img[x + i][y][1]
        W2 += img[x][y + i][1]
        W3 += img[x + i][y + i][1]
        W4 += img[x - i][y - i][1]
    W1 = W1 / size
    W2 = W2 / size
    W3 = W3 / size
    W4 = W4 / size
    Med = (max(W1, W2, W3, W4) + min(W1, W2, W3, W4) + img[x][y][1]) / 3
    Med = int(Med)
    img[x][y][0] = Med
    img[x][y][1] = Med
    img[x][y][2] = Med
    return img
def ShowImg(img):
    cv2.imshow("origin", img)
    cv2.waitKey(0)
def AFC1(img, size): #Adaptive Filter Combination, size : size of window
    (H, W, D) = img.shape
    for i in range(size, H - size):
        for j in range(size, W - size):
            img = MSM_filter(img, i, j, size)
    return img
def AFC2(img, size):
    return img
img = cv2.imread("image.png")
edge = cv2.Canny(img, 256, 256)
print(edge)