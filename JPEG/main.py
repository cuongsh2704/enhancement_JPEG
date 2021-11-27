import cv2
import numpy as np
def ShowImg(img1, img2):
    cv2.imshow("title", img1)
    cv2.imshow("title2", img2)
    cv2.waitKey(0)
def filterD(img, x, y, size): #hodges_lehman D,size : size of window , r : range , img(x,y)
    space = int(size/2)
    center = int(img[x][y][1])
    D = []
    for u in range(x - space, x + space + 1):
        for v in range(y - space, y + space + 1):
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
    for i in range(0, h + 1):
        for j in range(0, w + 1):
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
        W4 += img[x + i][y - i][1]
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

def NZC(img, x, y): # default window size = 5 , n = 12
    nzc = 0;
    Ti =[]
    for i in range(x - 2, x + 2):
        for j in range(y - 2, y + 2):
         Ti.append(img[i][j][1])

    Ti = np.mean(Ti)

    for i in range(x - 2, x + 2):
        for j in range(y - 2, y + 2):
         img[i][i][1] -= Ti
    return img

def RNZC(img, x, y):
    img = NZC(img, x, y)
    W1 = 0
    W2 = 0
    W3 = 0
    W4 = 0
    for i in range(-2, 2):
        if (img[x + i][y][1] * img[x + i + 1][y][1] < 0):
            W1 += 1
        if (img[x][y + i][1] * img[x][y + i + 1][1] < 0):
            W2 += 1
        if (img[x + i][y + i][1] * img[x + i + 1][y + i + 1][1] < 0):
            W3 += 1
        if (img[x + i][y - i][1] * img[x + i + 1][y - i - 1][1] < 0):
            W4 += 1
    return W1 + W2 + W3 + W4
def Classify(img): #classify pixel
    (H, W, D) = img.shape
    img_classify = cv2.Canny(img, 100, 100)
    for i in range(0, H):
        for j in range(0, W):
            rnzc = img_classify[i][j]
    return img_classify
def AFC1(img, size): #Adaptive Filter Combination, size : size of window
    img_classify = Classify(img)
    (H,W,D) = img.shape
    size = int(size / 2)
    for i in range(size, H - size):
        for j in range(size, W - size):
            if (img_classify[i][j] != 255):
                filterD(img, i, j, 3)
    return img
def AFC2(img, size):
    return img
def Evaluate(img):

    (H, W, D) = img.shape
    Med = np.mean(img)
    for i in range(0, H):
        for j in range(0, W):
            img[i][j][1] -= Med
            img[i][j][2] -= Med
            img[i][j][0] -= Med
    Med = np.mean(img)
    return Med
img = cv2.imread("image.png")
im = AFC1(img,5)
print(Evaluate(im), Evaluate(img))