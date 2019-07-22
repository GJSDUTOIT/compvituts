# pixels = image.load()

# corner = int(mask_size/2)
# print(corner)
# for i in range(corner, width):
#     for j in range(corner, height):
#         print(pixels[i,j])


import numpy
from PIL import Image, ImageOps, ImageChops
import cv2

def meanFilter(im, mask_size):
    img = im
    w = int(mask_size/2)
    iw = im.shape[0]
    ih = im.shape[1]
    for i in range(w,im.shape[0]-w):
        for j in range(w,im.shape[1]-w):
            block = im[i-w:i+w+1, j-w:j+w+1]
            m = numpy.mean(block,dtype=numpy.float32)
            img[i][j] = int(m)

    # img = numpy.zeros([im.shape[0]-w,im.shape[1]-w,3],dtype=np.uint8)
    # img.fill(255)
    block = img[w:iw-w, w:ih-w]
    return block

def medianFilter(im, mask_size):
    img = im
    w = int(mask_size/2)
    iw = im.shape[0]
    ih = im.shape[1]
    for i in range(w,im.shape[0]-w):
        for j in range(w,im.shape[1]-w):
            block = im[i-w:i+w+1, j-w:j+w+1]
            m = numpy.median(block)
            img[i][j] = int(m)

    # img = numpy.zeros([im.shape[0]-w,im.shape[1]-w,3],dtype=np.uint8)
    # img.fill(255)
    block = img[w:iw-w, w:ih-w]
    return block


def drive_mean(mask_size):
    w = int(mask_size/2)
    ImageOps.expand(Image.open('noisypears.tif'),border=w,fill='black').save('noisypears-border.tif')
    im = cv2.imread("noisypears-border.tif")
    im = meanFilter(im, mask_size)
    new_img = Image.fromarray( im, mode='RGB' )
    name = str(mask_size) + "_mean.tif"
    new_img.save(name)

def drive_median(mask_size):
    w = int(mask_size/2)
    ImageOps.expand(Image.open('noisypears.tif'),border=w,fill='black').save('noisypears-border.tif')
    im = cv2.imread("noisypears-border.tif")
    im = medianFilter(im, mask_size)
    new_img = Image.fromarray( im, mode='RGB' )
    name = str(mask_size) + "_median.tif"
    new_img.save(name)

drive_median(3)
drive_median(7)
drive_median(13)
