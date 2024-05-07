
# importing required libraries of opencv 
import cv2 
  
# importing library for plotting 
from matplotlib import pyplot as plt 
  
# reads an input image 
img = cv2.imread('/home/asreen-mohammad/Downloads/doo.jpeg',0) 
cv2.imwrite('/home/asreen-mohammad/Downloads/scripts/doraemon.jpeg', img )
 
# find frequency of pixels in range 0-255 
histr = cv2.calcHist([img],[0],None,[256],[0,256]) 
  
# show the plotting graph of an image 
plt.plot(histr) 
plt.show() 
