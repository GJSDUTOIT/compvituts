import numpy as np
from PIL import Image
import cv2
import math
import sys

def driver(s, theta, filename):
    image_data = cv2.imread(filename)
    m, n, d = image_data.shape
    print('original size')
    print(m,n)
    (x0, y0) = gen_center_c(m,n)
    H = gen_H(x0,y0,s,theta)
    warped_image_data = projectify(H,image_data,s,m,n)
    new_filename = 's' + str(s) + '_t' + str(round(math.degrees(theta))) + '.jpg'
    cv2.imwrite(new_filename, warped_image_data)
def gen_resulting_size(H,m,n):
    p1 = np.dot(H,[0, 0, 1])
    p1 = p1/p1[2];
    p2 = np.dot(H,[n, 0, 1])
    p2 = p2/p2[2]
    p3 = np.dot(H,[0, m, 1])
    p3 = p3/p3[2]
    p4 = np.dot(H,[n, m, 1])
    p4 = p4/p4[2]
    minx = math.floor(min([p1[0],p2[0],p3[0],p4[0]]));
    maxx = math.ceil(max([p1[0],p2[0],p3[0],p4[0]]));
    miny = math.floor(min([p1[1],p2[1],p3[1],p4[1]]));
    maxy = math.ceil(max([p1[1],p2[1],p3[1],p4[1]]));
    nn = maxx - minx + 1;
    mm = maxy - miny + 1;
    return (mm,nn)
def gen_center_c(m,n):
    return ((m-1)/2,(n-1)/2)
def gen_H(x0, y0, s, theta):
    alpha = s*math.cos(theta)
    beta = s*math.sin(theta)

    h00 = alpha
    h01 = beta
    h02 = (1 - alpha)*x0 - (beta)*y0
    h10 = -beta
    h11 = alpha
    h12 = beta*x0 + (1-alpha)*y0
    h20 = 0
    h21 = 0
    h22 = 1
    H = np.asarray([[h00,h01,h02],[h10,h11,h12],[h20,h21,h22]])
    return H

def projectify(H,x,s,m,n):
    print('resulting size')
    print(gen_resulting_size(H,m,n))
    warped_image_data = cv2.warpPerspective(x,H,gen_resulting_size(H,m,n))
    return warped_image_data

if __name__ == '__main__':
    filename = sys.argv[1]
    driver(1,math.radians(60), filename)
    driver(2,math.radians(90), filename)
    driver(0.5,math.radians(30), filename)
    driver(1.55,math.radians(315), filename)






# cv2.imshow("Poster", background)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
