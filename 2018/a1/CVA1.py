from PIL import Image

def cmpT(t1, t2):
  return sorted(t1) == sorted(t2)

img = Image.open('greenscreen.jpg')
bg = Image.open("bg.jpg")

new_size = (img.size[0],img.size[1])
new_im = Image.new("RGB", new_size)

bg.thumbnail((img.size[0]*1.5,img.size[1]*1.5))

pixels = img.load()
newpixels = new_im.load()
bgpixels = bg.load()

black = (0, 0, 0)
white = (255, 255, 255)
difx = bg.size[0] - img.size[0]
dify = bg.size[1] - img.size[1]

for i in range(img.size[0]):
    for j in range(img.size[1]):
        if pixels[i,j][1] > 127 and pixels[i,j][0] < 120 and pixels[i,j][2] < 150:
            pixels[i,j] = black

img.save("CVA1_1_step1.jpg","JPEG")
img.show()

for i in range(img.size[0]):
    for j in range(img.size[1]):
        if cmpT(pixels[i,j], black):
            newpixels[i,j] = black
        else:
            newpixels[i,j] = white
new_im.save("CVA1_1_step2.jpg","JPEG")
new_im.show()

for i in range(bg.size[0]):
    for j in range(bg.size[1]):
        if i > (difx) and j > (dify):
            if cmpT(pixels[i-(difx),j-(dify)], black):
                continue
            bgpixels[i,j] = pixels[i-(difx),j-(dify)]

bg.save("CVA1_1_step3.jpg","JPEG")
bg.show()
