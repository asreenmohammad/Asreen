## Histogram
1. Install Fowolling packages
   numpy, opencv, matplotlib

2. code
 ```bash
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
   ```
## carty
```

import os
import csv
from PIL import Image, ImageDraw


csv_file = "/home/asreen-mohammad/Downloads/7622202030987_bounding_box.csv"
image_dir = "/home/asreen-mohammad/Downloads/7622202030987/"
output_dir = "/home/asreen-mohammad/Downloads/7622202030987_with_boxes"


os.makedirs(output_dir, exist_ok=True)


def draw_boxes(image, boxes):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        left = int(box['left'])
        top = int(box['top'])
        right = int(box['right'])
        bottom = int(box['bottom'])
        draw.rectangle([left, top, right, bottom], outline="red")
    return image


def crop_image(image, boxes):
    cropped_images = []
    for box in boxes:
        left = int(box['left'])
        top = int(box['top'])
        right = int(box['right'])
        bottom = int(box['bottom'])
        cropped_img = image.crop((left, top, right, bottom))
        cropped_images.append(cropped_img)
    return cropped_images
```
    
## rough
```
num = list(range(10))
previousNum = 0
for i in num:
    sum = previousNum + i
    print('Current Number '+ str(i) + 'Previous Number ' + str(previousNum) + 'is ' + str(sum))
    previousNum=i
```

## vedio

```
num = list(range(10))
previousNum = 0
for i in num:
    sum = previousNum + i
    print('Current Number '+ str(i) + 'Previous Number ' + str(previousNum) + 'is ' + str(sum)) # <- This is the issue.
    previousNum=i
```


## video
```
#import the opencv library 
import cv2 
  
  
# define a video capture object 
vid = cv2.VideoCapture(0) 
  
while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
  
    # Display the resulting frame 
    cv2.imshow('frame', frame) 
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  # After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 
```




      
    
