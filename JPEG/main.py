import cv2
import numpy as np
def ShowImg(img1, img2):
    cv2.imshow("title", img1)
    cv2.imshow("title2", img2)
    cv2.waitKey(0)
def filterD(img, x, y, size, r):
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
img = cv2.imread("image.png")
(H, W, D) = img.shape
cv2.imshow("test1",img)
edge = cv2.Canny(img, 100, 100)
for i in range(5, H - 5):
    for j in range(5, W - 5):
        img = filterD(img, i, j, 5, 20)

cv2.imshow("test", img)
cv2.waitKey(0)