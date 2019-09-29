import cv2
from PIL import Image
import numpy
from scipy.spatial import distance
import math

def my_round(val, w, h, i):
    result = round(val)
    if i == 0 and result >= w:
        return (result - 1)
    elif i == 1 and result >= h:
        return (result - 1)
    else:
        return result
def bullseye(val):
    check  = math.floor(val) - val
    if check == 0:
        return True
    else:
        return False
def validate(val, w, h, i):
    if i == 0 and val >= w:
        return False
    elif i == 1 and val >= h:
        return False
    else:
        return True
def nneighbour(factor):
    img1 = cv2.imread("bird.jpg", cv2.IMREAD_COLOR)
    #factor = 0.5
    # make new image
    newimg = numpy.zeros([int(img1.shape[0]*factor),int(img1.shape[1]*factor),3],dtype=numpy.uint8)
    newimg.fill(255)

    w = img1.shape[0]
    h = img1.shape[1]
    for i in range(newimg.shape[0]):
        for j in range(newimg.shape[1]):
            newimg[i][j][0] = img1[my_round((i/factor),w,h,0)][my_round((j/factor),w,h,1)][0]
            newimg[i][j][1] = img1[my_round((i/factor),w,h,0)][my_round((j/factor),w,h,1)][1]
            newimg[i][j][2] = img1[my_round((i/factor),w,h,0)][my_round((j/factor),w,h,1)][2]


    name = str(factor) + "_nn.jpg"
    cv2.imwrite(name, newimg)
    #cv2.imshow('image',newimg)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
def bill(factor):
    img1 = cv2.imread("bird.jpg", cv2.IMREAD_COLOR)
    #factor = 0.5
    newimg = numpy.zeros([int(img1.shape[0]*factor),int(img1.shape[1]*factor),3],dtype=numpy.uint8)
    newimg.fill(255)
    w = img1.shape[0]
    h = img1.shape[1]
    for i in range(newimg.shape[0]):
        for j in range(newimg.shape[1]):
            pos_y = j/factor
            pos_x = i/factor
            if bullseye(pos_x) and bullseye(pos_y):
                newimg[i][j][0] = img1[int(pos_x)][int(pos_y)][0]
                newimg[i][j][1] = img1[int(pos_x)][int(pos_y)][1]
                newimg[i][j][2] = img1[int(pos_x)][int(pos_y)][2]
            else:
                t0 = (pos_x, pos_y)
                pix1_x = math.floor(pos_x)
                pix1_y = math.floor(pos_y)
                t1 = (pix1_x, pix1_y)
                # --------------------------
                pix2_y = math.ceil(pos_y)
                pix2_x = math.ceil(pos_x)
                if validate(pix2_x,w,h,0) != True or validate(pix2_y,w,h,1) != True:
                    pix2_x, pix2_y = 0, 0
                t2 = (pix2_x, pix2_y)
                # --------------------------
                pix3_x = math.floor(pos_x)
                pix3_y = math.ceil(pos_y)
                if validate(pix3_y,w,h,1) != True:
                    pix3_x, pix3_y = 0, 0
                t3 = (pix3_x, pix3_y)
                # --------------------------
                pix4_x = math.ceil(pos_x)
                pix4_y = math.floor(pos_y)
                t4 = (pix4_x, pix4_y)
                if validate(pix4_x,w,h,0) != True:
                    pix4_x, pix4_y = 0, 0
                # --------------------------
                dis_1 = distance.euclidean(t0, t1)
                dis_2 = distance.euclidean(t0, t2)
                dis_3 = distance.euclidean(t0, t3)
                dis_4 = distance.euclidean(t0, t4)
                tdis = dis_1 + dis_2 + dis_3 + dis_4
                tdis_1 = dis_1/tdis
                tdis_2 = dis_2/tdis
                if validate(pix2_y,w,h,1) != True or validate(pix2_x,w,h,0) != True:
                    tdis_2 = 0
                tdis_3 = dis_3/tdis
                if validate(pix3_y,w,h,1) != True:
                    tdis_3 = 0
                tdis_4 = dis_4/tdis
                if validate(pix4_x,w,h,0) != True:
                    tdis_4 = 0
                red = (img1[pix1_x][pix1_y][0]*tdis_1) + (img1[pix2_x][pix2_y][0]*tdis_2) + (img1[pix3_x][pix3_y][0]*tdis_3) + (img1[pix4_x][pix4_y][0]*tdis_4)
                green = (img1[pix1_x][pix1_y][1]*tdis_1) + (img1[pix2_x][pix2_y][1]*tdis_2) + (img1[pix3_x][pix3_y][1]*tdis_3) + (img1[pix4_x][pix4_y][1]*tdis_4)
                blue = (img1[pix1_x][pix1_y][2]*tdis_1) + (img1[pix2_x][pix2_y][2]*tdis_2) + (img1[pix3_x][pix3_y][2]*tdis_3) + (img1[pix4_x][pix4_y][2]*tdis_4)
                newimg[i][j][0] = red
                newimg[i][j][1] = green
                newimg[i][j][2] = blue

    name = str(factor) + "_bill.jpg"
    cv2.imwrite(name, newimg)
    #cv2.imshow('image',newimg)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

nneighbour(0.5)
print("\a")
bill(0.5)
print("\a")
nneighbour(2)
print("\a")
bill(2)
print("\a")
print("\a")
print("\a")

