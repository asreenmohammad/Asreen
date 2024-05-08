This code is a Python script that utilizes the OpenCV (Open Source Computer Vision) library and Matplotlib to read an image, calculate the histogram of pixel intensities, and then plot the histogram.

## Importing Libraries:
import cv2 
from matplotlib import pyplot as plt
## Reading the Input Image:
img = cv2.imread('/home/asreen-mohammad/Downloads/doo.jpeg',0)

![doo](https://github.com/asreenmohammad/Asreen/assets/169051643/d62a9c59-3648-487c-bf1d-3943251eb869)

## Saving the Image:
cv2.imwrite('/home/asreen-mohammad/Downloads/scripts/doraemon.jpeg', img)
## Calculating Histogram:
histr = cv2.calcHist([img],[0],None,[256],[0,256]) 
## Plotting the Histogram:
plt.plot(histr) 
plt.show()

![aashi](https://github.com/asreenmohammad/Asreen/assets/169051643/9f7c9f34-cedf-4f5e-9ac9-7434e1daf085)



    











## carty
This code processes images and their associated bounding box information from a CSV file. Let's break down the code step by step
## Imports
import os
import csv
from PIL import Image, ImageDraw
## Setting up File Paths:
csv_file = "/home/asreen-mohammad/Downloads/7622202030987_bounding_box.csv"
image_dir = "/home/asreen-mohammad/Downloads/7622202030987/"
output_dir = "/home/asreen-mohammad/Downloads/7622202030987_with_boxes"



      

    
## rough

This code calculates the running sum of numbers in the num list and prints each step of the calculation along with the current number, the previous number, and their sum.
input

## Initializing Variables:
num = list(range(10))
previousNum = 0
## Looping Through the Numbers:
for i in num:
## Calculating the Running Sum:
sum = previousNum + i
## Printing the Results:
print('Current Number '+ str(i) + 'Previous Number ' + str(previousNum) + 'is ' + str(sum))
## Updating the Previous Sum:
previousNum = i
    This line updates the value of previousNum to the current number i for the next iteration of the loop.
    However, there's an issue with this code. It's incorrectly updating previousNum with the current value of i in each iteration, effectively resetting it to the current number rather than accumulating the sum. To fix this and correctly calculate the running sum, you should add i to previousNum in each iteration:
output
```
Current Number 0Previous Number 0is 0

Current Number 1Previous Number 0is 1

Current Number 2Previous Number 1is 3

Current Number 3Previous Number 2is 5

Current Number 4Previous Number 3is 7

Current Number 5Previous Number 4is 9

Current Number 6Previous Number 5is 11

Current Number 7Previous Number 6is 13

Current Number 8Previous Number 7is 15

Current Number 9Previous Number 8is 17

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




      
    
