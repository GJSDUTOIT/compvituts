from cv2 import *
import numpy as np
from math import cos, sin, floor, ceil

def mmin(var1, var2, var3, var4):
    return min(min(var1, var2), min(var3, var4))
def mmax(var1, var2, var3, var4):
    return max(max(var1, var2), max(var3, var4))
def applyhomo(A, H):
    A = A.astype(float)
    (m, n, c) = image.shape[:3]
    p1 = np.dot(H,np.array([[1],[1],[1]]))
    p1 = p1/p1[2];
    p2 = np.dot(H,np.array([[n],[1],[1]]));
    p2 = p2/p2[2];
    p3 = np.dot(H,np.array([[1],[m],[1]]));
    p3 = p3/p3[2];
    p4 = np.dot(H,np.array([[n],[m],[1]]));
    p4 = p4/p4[2];
    minx = floor(mmin(p1[0], p2[0], p3[0], p4[0]))
    maxx = ceil(mmax(p1[0], p2[0], p3[0], p4[0]))
    miny = floor(mmin(p1[1], p2[1], p3[1], p4[1]))
    maxy = ceil(mmax(p1[1], p2[1], p3[1], p4[1]))
    nn = maxx - minx + 1
    mm = maxy - miny + 1
    print(minx,maxx,miny,maxy)
    B = np.zeros([mm,nn,c], dtype=np.uint8)
    B.fill(255)
    Hi = np.linalg.inv(H)
    for x in range(nn):
        for y in range(mm):
            p = [[x + minx -1], [y + miny - 1], [1]];
            pp = np.dot(Hi,p)
            xp = pp[0]/pp[2]
            yp = pp[1]/pp[2]
            xpf = floor(xp - 1); xpc = xpf + 1;
            ypf = floor(yp - 1); ypc = ypf + 1;
            print("x: ",x," y: ",y," p:", p)
            print("x: ",x," y: ",y," pp:", pp)
            print("x: ",x," y: ",y," xp:", xp)
            print("x: ",x," y: ",y," yp:", yp)
            print("x: ",x," y: ",y," xpf:", xpf)
            print("x: ",x," y: ",y," xpc:", xpc)
            print("x: ",x," y: ",y," ypf:", ypf)
            print("x: ",x," y: ",y," ypc:", ypc)
            if (xpf > 0) and (xpc <= n) and (ypf > 0) and (ypc <= m):
                # step1 = ((xpc - xp) * (ypc - yp) * A[ypf, xpf, :])
                # step2 = ((xpc - xp) * (yp - ypf) * A[ypc,xpf,:])
                # step3 = ((xp - xpf) * (ypc - yp) * A[ypf,xpc,:])
                # step4 = ((xp - xpf) * (yp - ypf) * A[ypc, xpc, :])
                # B[y,x,:] = step1 + step2 + step3 + step4
                continue

    B = B.astype(np.uint8)
    return B
image = cv2.imread("thbth.png", cv2.IMREAD_COLOR)
(m, n) = image.shape[:2]
cx = int((m-1)/2)
cy = int((n-1)/2)
theta = 45
s = 2
H = np.array([[(s*cos(theta)),(-s * sin(theta)),((cx*s*cos(theta))-(cy*s*sin(theta))-(cx))],[(s*sin(theta)),(s*cos(theta)),((cx*s*sin(theta))+(cy*s*cos(theta))-(cy))],[(0),(0),(1)]])
print(H)
B = applyhomo(image, H)

cv2.imshow("B", B)
cv2.waitKey(0)
cv2.destroyAllWindows()
