import cv2
import numpy as np
def ShowImg(img1, img2):
    cv2.imshow("title", img1)
    cv2.imshow("title2", img2)
    cv2.waitKey(0)
def flatImg(img):
    (h, w, d) = img.shape
    img_flat = np.zeros((h, w), dtype= int)
    for i in range(0, h):
        for j in range(0, w):
            img_flat[i][j] = img[i][j][0]
    return img_flat
def filterD(img, x, y, size): #hodges_lehman D,size : size of window , r : range , img(x,y)
    space = int(size/2)
    D = []
    for u in range(x - space, x + space + 1):
        for v in range(y - space, y + space + 1):
            D.append(img[u][v][0])
    D = sorted(D)
    Ave = []
    N = int(len(D) / 2)
    lenD = len(D)
    for i in range(0, N):
        value = (int(D[i]) + int(D[lenD - i -2])) / 2
        Ave.append(value)
    Med = np.mean(Ave)
    return Med
def median_filter(img, x, y, size):
    arr = []
    for i in range(x - 2, x + 2):
        for j in range(y - 2, y + 2):
            arr.append(img[i][j][1])
    Med = np.mean(arr)
    return Med

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
    return Med

def thresholding(img_window, img_zc): #LSD value n = 12
    n = 12

    med = np.mean(img_window);

    for i in range(0, 5):
        for j in range(0, 5):
            img_zc[i][j] = img_window[i][j] - med
            if (np.abs(img_zc[i][j]) < n) :
                img_window[i][j] = med;
                img_zc[i][j] = 0;



def RNZC(img, x, y): #size = 5 ,
    img_window = np.zeros((5, 5));
    img_zc = np.zeros((5, 5));
    for i in range(x - 2, x + 2):
        for j in range(y - 2, y + 2):
            img_window[i - x + 2][j - y + 2] = img[i][j][0];
            img_zc[i - x + 2][j - y + 2] = img_window[i - x + 2][j - y + 2]

    #thresholding:
    thresholding(img_window, img_zc)
    thresholding(img_window, img_zc)
    zc = 0
    for i in range(0, 5):
        for j in range(1, 5):
            if (img_zc[i][j] * img_zc[i][j - 1] < 0):
                zc += 1
            if (img_zc[j][i] * img_zc[j - 1][i] < 0):
                zc +=1

    for i in range(1, 5):
        if (img_zc[i][i] * img_zc[i - 1][i - 1] < 0):
            zc += 1
        if (img_zc[i][4 - i] * img_zc[i - 1][4 - i] < 0):
            zc += 1

    return zc
def Classify(img): #classify pixel
    (H, W, D) = img.shape
    b = 12
    img_classify = cv2.Canny(img, 100, 100)
    for i in range(0, H - 2):
        for j in range(0, W - 2):
            if (img_classify[i][j] != 255 & RNZC(img, i, j) > b):
                img_classify[i][j] = 128
    return img_classify
def AFC1(img, size): #Adaptive Filter Combination, size : size of window
    img_classify = Classify(img)
    (H,W,D) = img.shape
    size = int(size / 2)
    for i in range(size, H - size):
        for j in range(size, W - size):
            if (img_classify[i][j] == 255):
                for k in range(0, 3):
                    img[i][j][k] = median_filter(img, i, j, size)
            if (img_classify[i][j] == 0):
                for k in range(0, 3):
                    img[i][j][k] = filterD(img, i, j, 3);
def AFC2(img, size):
    img_classify = Classify(img)
    (H, W, D) = img.shape
    size = int(size / 2)
    for i in range(size, H - size):
        for j in range(size, W - size):
            if (img_classify[i][j] == 0):
                for k in range(0, 3):
                    img[i][j][k] = filterD(img, i, j, 3);
    for i in range(size, H - size):
        for j in range(size, W - size):
            if (img_classify[i][j] == 255):
                for k in range(0, 3):
                    img[i][j][k] = MSM_filter(img, i, j, 5);
    for i in range(size, H - size):
        for j in range(size, W - size):
            for k in range(0, 3):
                img[i][j][k] = filterD(img, i, j, 3);

def Evaluate(img, img_fil):
    (H, W, D) = img.shape
    med = []
    for i in range(0, H):
        for j in range(0, W):
            med.append(np.abs(int(img[i][j][0]) - int(img_fil[i][j][0])))

    Med_eval = np.mean(med)
    return Med_eval

img = cv2.imread("image.png")
img_jpeg = cv2.imread("anh.png")
AFC2(img_jpeg, 5)

cv2.imshow("filter", img_jpeg)
cv2.waitKey(0)







