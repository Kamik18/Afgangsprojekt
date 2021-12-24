# %%
# import and method def
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from matplotlib import cm
import time
from glob import glob

def printBeforeAndAfter(subtitle_left, subtitle_right, before, after):
    fig = plt.figure(figsize=(20,15))
    rows = 1
    columns = 2
    fig.add_subplot(rows,columns, 1)
    plt.title(subtitle_left)
    plt.imshow(before)
    fig.add_subplot(rows,columns, 2)
    plt.title(subtitle_right)
    plt.imshow(after)
# %%
img_dir = 'RobotWorkspace'
num = 0
img = cv2.imread(f'{img_dir}/Robot_Workspace_{num}.jpg')
# Used to display images with pillow.
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_with_line = img.copy()
rows, cols = img_with_line.shape[:2]
y = 470
x_left = 188
x_right = 565
left = (x_left, y)
right = (x_right, y)
width = x_left - x_right
print(width)
cv2.line(img_with_line, left, right, (255,0,0), 2)

printBeforeAndAfter('Original', 'Line', img, img_with_line)
# %%
