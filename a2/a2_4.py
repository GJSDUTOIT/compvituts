from PIL import Image, ImageDraw
import numpy as np
import sys

def driver(filename):
    template = Image.open("template.jpg")
    # template_data = np.asarray(template)
    distortion = Image.open(filename)
    corner_paste(distortion,template)
    pmatrix = read_points(filename)
    draw_lines(distortion,pmatrix)
    distortion.save(gen_new_filename(filename))
    distortion.show()

def corner_paste(distortion, template):
    template_data = np.asarray(template)
    mt, nt, ct = template_data.shape
    distortion.paste(template, (0,0))

def read_points(filename):
    data_filename = get_data_filename(filename)
    pmatrix = np.loadtxt(data_filename)
    return pmatrix

def get_data_filename(filename):
    splitstringarray = filename.split('.')
    data_filename = './fifasiftmatches/siftmatches_' + splitstringarray[0] + '.txt'
    return data_filename
def gen_new_filename(filename):
    new_filename = 'matched_' + filename
    return new_filename
def draw_lines(distortion, pmatrix):
    draw = ImageDraw.Draw(distortion)
    m = pmatrix.shape[0]
    for r in range(m):
        draw.line((pmatrix[r,0],pmatrix[r,1],pmatrix[r,2],pmatrix[r,3]), fill=128, width=3)

if __name__=='__main__':
    filename = sys.argv[1]
    driver('1.jpg')
    driver('2.jpg')
    driver('3.jpg')
    driver('4.jpg')
    driver('5.jpg')
    driver('6.jpg')
    driver('7.jpg')
    driver('8.jpg')
    driver('9.jpg')
    driver('10.jpg')
    driver('11.jpg')
    driver('12.jpg')
