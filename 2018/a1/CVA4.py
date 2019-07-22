from cv2 import *
import numpy as np

img1 = None
img2 = None
orb = None

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

def set_im1_name(name):
    global img1
    img1 = cv2.imread(name, 0)
    img1 = image_resize(img1, width=800)
def set_im2_name(name):
    global img2
    img2 = cv2.imread(name, 0)
    img2 = image_resize(img2, width=800)
def init():
    global orb
    orb = cv2.ORB_create()
def get_coordinates_descriptors_tuple_img1():
    global orb
    if orb is None:
        orb = cv2.ORB_create()
    return orb.detectAndCompute(img1, None)
def get_coordinates_descriptors_tuple_img2():
    global orb
    if orb is None:
        orb = cv2.ORB_create()
    return orb.detectAndCompute(img2, None)
def match_and_display_images(threshold=50):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    kp1, des1 = get_coordinates_descriptors_tuple_img1()
    kp2, des2 = get_coordinates_descriptors_tuple_img2()
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)
    matching_result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:threshold], None, flags=2)
    cv2.imshow("Matching result", matching_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def example1():
    set_im1_name("semper1.jpg")
    set_im2_name("semper2.jpg")
    init()
    match_and_display_images(40)
def example2():
    set_im1_name("twin1.jpg")
    set_im2_name("twin2.jpg")
    init()
    match_and_display_images()
example1()
