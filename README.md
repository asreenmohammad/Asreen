## pic

## HISTOGRAM

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

## BOUNDING BOX


This code processes images and their associated bounding box information from a CSV file. Let's break down the code step by step
## Imports
import os
import csv
from PIL import Image, ImageDraw
## Setting up File Paths:
csv_file = "/home/asreen-mohammad/Downloads/7622202030987_bounding_box.csv"

![7622202030987_f306535d741c9148dc458acbbc887243_L_538](https://github.com/asreenmohammad/Asreen/assets/169051643/689e51ef-4712-4157-a79f-0ad043f47acc)

## Helper Functions:

    draw_boxes(image, boxes): This function draws bounding boxes on the input image.
    crop_image(image, boxes): This function crops the input image based on the bounding box coordinates and returns a list of cropped images.
    
## Reading and Processing CSV File:

with open(csv_file, 'r') as file:

    csv_reader = csv.DictReader(file)
    
    for row in csv_reader:
    
        image_name = row['filename']
        
        image_path = os.path.join(image_dir, image_name)
        
        output_path = os.path.join(output_dir, image_name)
        
        image = Image.open(image_path)
        
        boxes = [{'left': row['xmin'], 'top': row['ymin'], 'right': row['xmax'], 'bottom': row['ymax']}]
        
        
## Processing Images:

cropped_images = crop_image(image, boxes)

        for i, cropped_img in enumerate(cropped_images):
        
            cropped_img.save(os.path.join(output_dir, f"{i}_{image_name}"))
            
        full_image_with_boxes = draw_boxes(image, boxes)
        
        full_image_with_boxes.save(os.path.join(output_dir, f"full_{image_name}"))
        
## Creating Output Directory:

os.makedirs(output_dir, exist_ok=True)

        
## output

image_dir = "/home/asreen-mohammad/Downloads/7622202030987/"


output_dir = "/home/asreen-mohammad/Downloads/7622202030987_with_boxes"



![full_7622202030987_f306535d741c9148dc458acbbc887243_L_495](https://github.com/asreenmohammad/Asreen/assets/169051643/04586b56-c529-4f54-bbad-9176bc5510ac)



![0_7622202030987_f306535d741c9148dc458acbbc887243_L_487](https://github.com/asreenmohammad/Asreen/assets/169051643/31516d5d-cb5f-43f2-a0b1-52e080d5c98f)


      

    
## rough

## PRINTING NUMDERS

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
    
   
## output

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




## vedio

## VIDEO

This Python code utilizes the OpenCV library to capture video from a camera device and save it to a file named "camera.avi". Let's break down the code step by step:

## Import the OpenCV library:
import cv2
## Create a VideoCapture object to capture video from the camera:
video = cv2.VideoCapture(0)
## Check if the camera is opened successfully:
if (video.isOpened() == False):
    print("Error reading video file")
## Get the frame width and height from the captured video:
frame_width = int(video.get(3))
frame_height = int(video.get(4))
## Define the size of the frames:
size = (frame_width, frame_height)
## Create a VideoWriter object to write video frames to a file:
result = cv2.VideoWriter('camera.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)
## Enter a loop to continuously capture frames from the camera:
while(True):
    ret, frame = video.read()
## If a frame is successfully captured:
if cv2.waitKey(1) & 0xFF == ord('s'):
    break
## Release the VideoCapture and VideoWriter objects after the loop ends:
video.release()
result.release()
## Close all OpenCV windows:
cv2.destroyAllWindows()
## Print a success message:
print("The video was successfully saved")

## output

https://github.com/asreenmohammad/Asreen/assets/169051643/598c5123-2a63-4a5a-9119-4a482ec787a7




      
    
