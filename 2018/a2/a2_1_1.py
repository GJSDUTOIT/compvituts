import cv2
import numpy as np
import math
import a2_1 as gen

def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

image = cv2.imread('small.png')
# cv2.imshow('awe',image)
# cv2.waitKey(0)
h, w = image.shape[:2]
image = image_resize(image, width=int(w*0.6))
image = rotateImage(image, -315)
cv2.imwrite('s0.6_t315.jpg', image)
cv2.imshow('awe', image)
cv2.waitKey(0)
