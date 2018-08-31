import numpy as np
from PIL import Image
def driver():
    poster =  Image.open("wss.jpg")
    background = Image.open("griest.jpg")
    posterdata = np.asarray(poster)
    backgrounddata = np.asarray(background)
    projectify(posterdata,gen_H())
def p(a,b):
    return (-a * b)
def gen_H():
    # x1,y1,x2,y2,x3,y3,x4,y4,x1p,y1p,x2p,y2p,x3p,y3p,x4p,y4p
    x1, y1 = 0, 0
    x2, y2 = 619, 0
    x3, y3 = 0, 919
    x4, y4 = 619,919
    x1p, y1p = 108, 250
    x2p, y2p = 310, 130
    x3p, y3p = 40, 610
    x4p, y4p = 305, 555
    A = np.matrix([[x1,y1,1,0,0,0,p(x1p,x1),p(x1p,y1),-x1p],[0,0,0,x1,y1,1,p(y1p,x1),p(y1p,y1),-y1p],[x2,y2,1,0,0,0,p(x2p,x2),p(x2p,y2),-x2p],[0,0,0,x2,y2,1,p(y2p,x2),p(y2p,y2),-y2p],[x3,y3,1,0,0,0,p(x3p,x3),p(x3p,y3),-x3p],[0,0,0,x3,y3,1,p(y3p,x3),p(y3p,y3),-y3p],[x4,y4,1,0,0,0,p(x4p,x4),p(x4p,y4),-x4p],[0,0,0,x4,y4,1,p(y4p,x4),p(y4p,y4),-y4p]])
    u,s,vh = np.linalg.svd(A)
    v = np.transpose(vh)
    col = v.shape[1] - 1
    hh = v[:,col]
    H = np.zeros((3,3))
    counter = 0
    for i in range(3):
        for j in range(3):
            H[i][j] = hh[counter]
            counter += 1
    return H
def car_to_hom(car_vector):
    return np.append(np.asarray(car_vector, dtype=np.float32).copy(), 1)
def hom_to_car(hom_vector):
    return np.asarray(hom_vector[:-1]/hom_vector[-1], dtype=np.float32)
def hnorm(hom_vector):
    return np.asarray(hom_vector[:]/hom_vector[-1], dtype=np.float32)
def projectify(A,H):
    # cast input image to double
    A = np.asarray(A, dtype=np.float32)
    H = np.asarray(H, dtype=np.float32)
    # determine size of rows, cols and colour channels
    m, n, c = A.shape
    # determine size of output image by forward-transforming four corners of A
    p1 = hnorm(np.dot(H,np.asarray([1,1,1], dtype=np.float32)))
    p2 = hnorm(np.dot(H,np.asarray([n,1,1], dtype=np.float32)))
    p3 = hnorm(np.dot(H,np.asarray([1,m,1], dtype=np.float32)))
    p4 = hnorm(np.dot(H,np.asarray([n,m,1], dtype=np.float32)))
    minx = np.floor(np.min([p1[0],p2[0],p3[0],p4[0]]))
    maxx = np.ceil(np.max([p1[0],p2[0],p3[0],p4[0]]))
    miny = np.floor(np.min([p1[1],p2[1],p3[1],p4[1]]))
    maxy = np.ceil(np.max([p1[1],p2[1],p3[1],p4[1]]))
    nn = maxx - minx
    mm = maxy - miny
    # initialize output with white pixels
    B = np.zeros((int(mm),int(nn),int(c)))
    B.fill(250)
    # precompute the inverse of H
    Hi = np.linalg.inv(H)
    # loop through B's pixels
    for x in range(int(nn)):
        for y in range(int(mm)):
            # compensate for shift in B's origin
            p = np.asarray([x + minx, y + miny, 1])
            # apply inverse of H
            pp = hom_to_car(np.dot(Hi,p))
            xp = pp[0]
            yp = pp[1]
            # perform bilinear interpolation
            xpf = np.floor(xp)
            xpc = xpf + 1
            ypf = np.floor(yp)
            ypc = ypf + 1
            if (xpf > 0) and (xpc <= n) and (ypf > 0) and (ypc <= m):
                B[y,x,:] = np.dot(((xpc - xp)*(ypc - yp)),A[ypf,xpf,:]) + np.dot(((xpc - xp)*(yp - ypf)),A[ypc,xpf,:]) + np.dot(((xp - xpf)*(ypc - yp)),A[ypf,xpc,:]) + np.dot(((xp - xpf)*(yp - ypf)),A[ypc,xpc,:])

    B = np.asarray(B, dtype=np.int8)
if __name__ == '__main__':
    driver()







# cv2.imshow("Poster", background)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
