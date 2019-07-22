from PIL import Image

old_im = Image.open('noisypears.tif')
old_im.show()
old_size = (old_im.size[0],old_im.size[1])

new_size = (old_im.size[0]+2,old_im.size[1]+2)
new_im = Image.new("L", new_size)
new_im.show()

print(new_size)
print(old_size)

newpix = new_im.load()
oldpix = old_im.load()

img_w, img_h = old_im.size
bg_w, bg_h = new_im.size
offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
new_im.paste(old_im, offset)
new_im.save('out.png')


new_im.show()
