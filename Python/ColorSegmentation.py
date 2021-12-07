#%%
# Import image
from PIL import Image
from matplotlib.pyplot import imshow
import cv2
from numpy.lib.type_check import imag
import numpy as np
import os
import glob
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def filelist(root, file_type):
    """Returns a fully-qualified list of filenames under root directory"""
    return [os.path.join(directory_path, f) for directory_path, directory_name, 
            files in os.walk(root) for f in files if f.endswith(file_type)]

def generate_train_df (anno_path):
    annotations = filelist(anno_path, '.xml')
    anno_list = []
    for anno_path in annotations:
        root = ET.parse(anno_path).getroot()
        objects = []
        for child in root:
            if child.tag == 'object':
                objects.append(child)

        anno = {}
        anno['filename'] = root.find("./filename").text
        anno['width'] = root.find("./size/width").text
        anno['height'] = root.find("./size/height").text
        
        bndboxes = []
        i = 0
        for objct in objects:
            i += 1   
            bndbox = {}
            bndbox['class'] = 0
            bndbox['xmin'] = int(objct.find("./bndbox/xmin").text)
            bndbox['ymin'] = int(objct.find("./bndbox/ymin").text)
            bndbox['xmax'] = int(objct.find("./bndbox/xmax").text)
            bndbox['ymax'] = int(objct.find("./bndbox/ymax").text)
            bndboxes.append(bndbox)
            
        anno['bndbox'] = bndboxes            
        anno_list.append(anno)
    return pd.DataFrame(anno_list)

img_path = 'data/AlfaLavalFinal/images'
anno_path = 'data/AlfaLavalFinal/annotations'

img_num = '122'
imgs = []
#os.chdir(img_path)
#for file in glob.glob("*.jpg"):   
#    imgs.append(file)
im_data = generate_train_df(anno_path)    
im_data.head()

frame = cv2.imread(f'{img_path}/Alfa_Laval_sensor_020.jpg', cv2.IMREAD_COLOR)
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
plt.imshow(frame)

#%%
#print(imgs[1])      
#imgs = [f'Alfa_Laval_Sensor_{img_num}.jpg']#, 'Alfa_Laval_Sensor_026.jpg',]
for idx, row in im_data.iterrows():
#if True:
    frame = cv2.imread(f'{img_path}/{row[0]}', cv2.IMREAD_COLOR)
    
    #frame = cv2.resize(frame, (447,300), interpolation=cv2.INTER_AREA)

    # Converts images from BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([105,90,0]) # 105, 90, 0
    upper_blue = np.array([135,255,255]) # 135, 255, 255

    # Here we are defining range of bluecolor in HSV
    # This creates a mask of blue coloured
    # objects found in the frame.
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(mask, kernel)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #print(len(contours))
    for contour in contours:
        if len(contour) < 10:
            cv2.drawContours(mask,  [contour], -1, color=0, thickness=-1)
        
            # Create Hough circles:
    
        #ellipse = cv2.fitEllipse(contour)
        #cv2.ellipse(frame, ellipse, (0,0,255), 2)
    
    xmin = row[3][0]['xmin']
    ymin = row[3][0]['ymin']
    xmax = row[3][0]['xmax']
    ymax = row[3][0]['ymax']
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
    # The bitwise and of the frame and mask is done so
    # that only the blue coloured objects are highlighted
    # and stored in res
    res = cv2.bitwise_and(frame,frame, mask= mask)
    
    # Convert colors for frame and mask
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    new_name = row[0].replace('.jpg', '_seg.jpg')
    #cv2.imwrite(f'data/AlfaLavalFinal/colorsegmented/{new_name}', res)
    
    #"""
    fig = plt.figure(figsize=(20, 15))
    #fig.suptitle(row[0])
    rows = 1
    columns = 3
    fig.add_subplot(rows, columns, 1)
    plt.imshow(frame)
    fig.add_subplot(rows, columns, 2)
    plt.imshow(mask)
    fig.add_subplot(rows, columns, 3)
    plt.imshow(res)
    if idx == 20:
        print("nr 20")

    #"""
#%%
"""
for img in imgs:
#if True:
    path = (f'{img}') 
    frame = cv2.imread(path, cv2.IMREAD_COLOR)
    #frame = cv2.resize(frame, (447,300), interpolation=cv2.INTER_AREA)

    # Converts images from BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([105,90,0]) # 105, 90, 0
    upper_blue = np.array([135,255,255]) # 135, 255, 255

    # Here we are defining range of bluecolor in HSV
    # This creates a mask of blue coloured
    # objects found in the frame.
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(mask, kernel)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print(len(contours))
    for contour in contours:
        if len(contour) < 10:
            cv2.drawContours(mask,  [contour], -1, color=0, thickness=-1)
        
            # Create Hough circles:
    
        #ellipse = cv2.fitEllipse(contour)
        #cv2.ellipse(frame, ellipse, (0,0,255), 2)
        
    
    # The bitwise and of the frame and mask is done so
    # that only the blue coloured objects are highlighted
    # and stored in res
    res = cv2.bitwise_and(frame,frame, mask= mask)
    cv2.imshow(f'{img}',frame)
    cv2.moveWindow(f'{img}', 100,100)
    cv2.imshow('mask',mask)
    cv2.moveWindow('mask', 700,100)
    cv2.imshow('res',res)
    cv2.moveWindow('res', 1300,100)


    cv2.waitKey(0) 
    # Destroys all of the HighGUI windows.
    cv2.destroyAllWindows()

"""