# import and method def
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from matplotlib import cm
import time
from glob import glob
import os

# Draw closest point
def createDot(img, point):
    for i in range(-5, 6, 1):
        for j in range(-5, 6, 1):
            x = point[1] + i
            y = point[0] + j
            if x < w and y < h:
                if len(img[y,x]) == 4:
                    img[y, x] = [255,0,0, 255]  
                if len(img[y,x]) == 3:
                    img[y, x] = [255,0,0]    

number = 301

#img_dir = './ObstacleAvoidance/images/*'
#num = '301'
#images = glob(img_dir)
img = cv2.imread('ObstacleAvoidance/images/Alfa_Laval_Sensor_301.jpg')
#for img in images:
if True:
    start = time.time()
    #img = cv2.imread(img)
    # Convert image color and blur image
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    medianBlur = cv2.medianBlur(img, 5)
    # Edge detection
    median_edges = cv2.Canny(medianBlur, 50, 130)

    # Sidefill
    h, w = img.shape[:2]
    row_inds = np.indices((h, w))[0] # gives row indices in shape of img
    row_inds_at_edges = row_inds.copy()
    row_inds_at_edges[median_edges==0] = 0 # only get indices at edges, 0 elsewhere
    max_row_inds = np.amax(row_inds_at_edges, axis=0) # find the max row ind over each col

    inds_after_edges = row_inds >= max_row_inds

    filled_from_bottom = np.zeros((h, w))
    filled_from_bottom[inds_after_edges] = 255

    cv2.imwrite(f'./ObstacleAvoidance/test/Alfa_Laval_Sensor_Fill_From_Bottom_{number}.jpg', filled_from_bottom)
    cv2.imshow('Detection', filled_from_bottom)
    cv2.waitKey(0)
    # Horizontal dialate and erode
    # Copy image
    horizontal = filled_from_bottom.copy()

    # Horizontal erode
    horizontal_size_erode = w // 30
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size_erode, 1))
    horizontal = cv2.erode(horizontal, horizontalStructure)

    # Horizontal dilate
    # This is done to remove small corns found in the floor
    horizontal_size_dialate = w // 20
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size_dialate, 1))
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    # 2nd horizontal erode 
    horizontal = cv2.erode(horizontal, horizontalStructure)

    cv2.imwrite(f'./ObstacleAvoidance/test/Alfa_Laval_Sensor_Horizontal_{number}.jpg', horizontal)
    cv2.imshow('Detection', horizontal)
    cv2.waitKey(0)

    # Smoothed
    converted_img = Image.fromarray(np.uint8(cm.gist_earth(horizontal)*255))
    smoothed = converted_img.filter(ImageFilter.ModeFilter(size=13))

    # Find furthest point 
    img_with_alpha_values = np.asarray(smoothed)
    cv2.imwrite(f'./ObstacleAvoidance/test/Alfa_Laval_Sensor_Smoothed_{number}.jpg', img_with_alpha_values)
    cv2.imshow('Detection', img_with_alpha_values)
    cv2.waitKey(0)
    # Set robot center point
    #createDot(img_with_alpha_values, (int(h*0.9), int(w/2)))

    # Convert image to grayscale and threshold it
    bw = cv2.cvtColor(img_with_alpha_values, cv2.COLOR_RGB2GRAY)
    bw = cv2.inRange(bw, 190, 255)

    # Find all occurences of white pixels
    pixel_y, pixel_x = np.nonzero(bw)
    createDot(img_with_alpha_values, (pixel_y[0], pixel_x[0]))

    cv2.imwrite(f'./ObstacleAvoidance/test/Alfa_Laval_Sensor_Dot_{number}.jpg', img_with_alpha_values)
    cv2.imshow('Detection', img_with_alpha_values)
    cv2.waitKey(0)

    # Construct a colour image to superimpose
    mask = np.asarray(img_with_alpha_values)

    img_with_overlay = cv2.addWeighted(img, 0.9, mask[:,:,:3], 0.3, 0)
    createDot(img_with_overlay, (pixel_y[0], pixel_x[0]))
    # end time
    end = time.time()
    print(f"Runtime of the program is {end - start}")
    cv2.imshow('Detection', img_with_overlay)
    cv2.waitKey(0)
    
    #cv2.imwrite(f'./ObstacleAvoidance/test/Alfa_Laval_Sensor_Test_With_Overlay_{number}.jpg', img_with_overlay)
    number += 1