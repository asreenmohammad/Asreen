import numpy as np
import cv2
from matplotlib import pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image', help='enter the image path')
parser.add_argument('--output', help='enter the output path')

args = vars(parser.parse_args())
image = cv2.imread(args['image'])

 
<<<<<<< HEAD
#img = cv.imread('doo.jpeg')
#cv.imwrite("/home/asreen-mohammad/Downloads/doraemon.jpeg",img)
=======
#img = cv.imread('lotus.jpeg')
#cv.imwrite("/home/bhumika-avadutha/Desktop/programs/b.jpeg",img)
>>>>>>> 3af4647... add new code
assert image is not None, "file could not be read, check with os.path.exists()"
color = ('b','g','r')
for i,col in enumerate(color):
 histr = cv2.calcHist([image],[i],None,[256],[0,256])
 plt.plot(histr,color = col)
 plt.xlim([0,256])
<<<<<<< HEAD
plt.show()
=======
plt.show()
>>>>>>> 3af4647... add new code
