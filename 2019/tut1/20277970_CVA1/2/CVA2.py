import numpy, cv2, math
from PIL import Image, ImageOps, ImageChops

def unsharp_filter(filename):
    #Modify image to be grayscale and save
    im = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
    cv2.imwrite('grayscale.jpg',im)
    #Apply smoothing filter & open original
    smoothed = (medianFilter('grayscale.jpg', 5)).astype(float)
    #smoothed = (cv2.imread('smoothed.jpg',cv2.IMREAD_GRAYSCALE)).astype(float)
    original = (cv2.imread('grayscale.jpg',cv2.IMREAD_GRAYSCALE)).astype(float)
    #Calculate mask
    sharpened = numpy.zeros(shape=[original.shape[0], original.shape[1]], dtype=numpy.float32)
    for i in range(original.shape[0]):
        for j in range(original.shape[1]):
                sharpened[i][j] = original[i][j] + (original[i][j] - smoothed[i][j])
    min_pixel = numpy.amin(sharpened)
    max_pixel = numpy.amax(sharpened)
    for i in range(sharpened.shape[0]):
        for j in range(sharpened.shape[1]):
            sharpened[i][j] = (((255.0-0.0)*(sharpened[i][j]-min_pixel))/(max_pixel - min_pixel)) + (0)
    sharpened = sharpened.astype(int)
    original = original.astype(int)
    smoothed = smoothed.astype(int)
    cv2.imwrite('original.jpg',original)
    cv2.imwrite('smoothed.jpg',smoothed)
    cv2.imwrite('sharpened.jpg',sharpened)

def medianFilter(filename, mask_size):
    #Add black border
    w = int(mask_size/2)
    pil_image = (ImageOps.expand(Image.open(filename),border=w,fill='black')).convert('L')
    im = numpy.array(pil_image)
    img = im
    #Apply median filter
    iw = im.shape[0]
    ih = im.shape[1]
    for i in range(w,im.shape[0]-w):
        for j in range(w,im.shape[1]-w):
            block = im[i-w:i+w+1, j-w:j+w+1]
            m = numpy.median(block)
            img[i][j] = int(m)

    #cut off black border
    block = img[w:iw-w, w:ih-w]
    return block
    

unsharp_filter("colour.jpg")


