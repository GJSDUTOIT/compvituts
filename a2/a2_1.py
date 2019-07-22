import numpy as np
from PIL import Image
import cv2
import math
import sys

def driver(s, theta, filename):
    image = Image.open(filename)
    #image.show()
    image_data = np.asarray(image)
    m, n = image_data.shape
    (x0, y0) = gen_center_c(m,n)
    print(x0,y0)
    H = gen_H(x0,y0,s,theta)
    warped_image_data = projectify(H,image_data)
    warped_image = Image.fromarray(warped_image_data)
    warped_image.show()

def gen_center_c(m,n):
    return ((m-1)/2,(n-1)/2)
def gen_H(x0, y0, s, theta):
    h00 = s * math.cos(theta)
    h01 = -s * math.sin(theta)
    h02 = (x0 * s * math.cos(theta)) - (y0 * s * math.sin(theta)) - x0
    h10 = s * math.sin(theta)
    h11 = s * math.cos(theta)
    h12 = (x0 * s * math.sin(theta)) + (y0 * s * math.cos(theta)) - y0
    h20 = 0
    h21 = 0
    h22 = 1
    H = np.asarray([[h00,h01,h02],[h10,h11,h12],[h20,h21,h22]])
    return H

def projectify(H,x):
    warped_image_data = cv2.warpPerspective(x,H,(x.shape[1], x.shape[0]))
    return warped_image_data

if __name__ == '__main__':
    filename = sys.argv[1]
    driver(1,math.radians(60), filename)







# cv2.imshow("Poster", background)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
