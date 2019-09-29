import numpy as np
from PIL import Image
import cv2

def driver():
    poster =  Image.open("wss.jpg")
    posterdata = np.asarray(poster)
    backgrounddata = np.array(Image.open("griest.jpg"))
    warped_poster_data = projectify(posterdata,backgrounddata,gen_H())
    warped_poster_image = Image.fromarray(warped_poster_data)
    warped_poster_image.save("warped_poster.jpg")
    output_data = overlay(warped_poster_data, backgrounddata)
    output = Image.fromarray(output_data)
    output.show()
    output.save("poster.jpg")
def p(a,b):
    return (-a * b)

def gen_H():
    # x1,y1,x2,y2,x3,y3,x4,y4,x1p,y1p,x2p,y2p,x3p,y3p,x4p,y4p
    x1, y1 = 0, 0
    x2, y2 = 619, 0
    x3, y3 = 0, 919
    x4, y4 = 619,919
    x1p, y1p = 107, 248
    x2p, y2p = 313, 126
    x3p, y3p = 38, 618
    x4p, y4p = 310, 560
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

def projectify(A,B,H):
    m, n, c = B.shape
    warped_poster_data = cv2.warpPerspective(A,H,(n,m))
    return warped_poster_data

def overlay(fg, bg):
    m, n, c = bg.shape
    #bg.setflags(write=1)
    for x in range(n):
        for y in range(m):
            #print(np.array_equal(fg[y,x,:],bg[y,x,:]))
            if np.array_equal(fg[y,x,:],[0,0,0]) == False:
                bg[y,x,:] = fg[y,x,:]

    return bg

if __name__ == '__main__':
    driver()







# cv2.imshow("Poster", background)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
