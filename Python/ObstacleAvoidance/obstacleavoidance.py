#%%
# import and method def
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from matplotlib import cm

def printBeforeAndAfter(title, before, after):
    fig = plt.figure(figsize=(20,15))
    fig.suptitle(title, fontsize=16)
    fig.subplots_adjust(top=1.4)
    rows = 1
    columns = 2
    fig.add_subplot(rows,columns, 1)
    plt.imshow(before)
    fig.add_subplot(rows,columns, 2)
    plt.imshow(after)
    
#%%
# # Edge detection - Canny
img_dir = '../data/Background'
num = '109'

#img = cv2.imread(f'{img_dir}/Alfa_Laval_Sensor_{num}.jpg')
img = cv2.imread('obstacle_scene_1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

edges = cv2.Canny(img, 100,200)
printBeforeAndAfter('No blur', img, edges)

blurred_img = cv2.GaussianBlur(img, (3,3),cv2.BORDER_DEFAULT)
blurred_edge = cv2.Canny(blurred_img, 100,200)
printBeforeAndAfter('Gaussianblur image', blurred_img, blurred_edge)

#%%
# Sidefill
h, w = img.shape[:2]
row_inds = np.indices((h, w))[0] # gives row indices in shape of img
row_inds_at_edges = row_inds.copy()
row_inds_at_edges[edges==0] = 0 # only get indices at edges, 0 elsewhere
max_row_inds = np.amax(row_inds_at_edges, axis=0) # find the max row ind over each col

inds_after_edges = row_inds >= max_row_inds

filled_from_bottom = np.zeros((h, w))
filled_from_bottom[inds_after_edges] = 255

printBeforeAndAfter('Sidefill', edges, filled_from_bottom)
#%%
# Horizontal erode
horizontal = filled_from_bottom
horizontal_size = w // 6
horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
horizontal = cv2.erode(horizontal, horizontalStructure)
printBeforeAndAfter('horizontal erode', filled_from_bottom, horizontal)
#%%
# Smoothed
from PIL import Image, ImageFilter, ImageDraw
from matplotlib import cm
#horizontal = np.uint8(horizontal)
converted_img = Image.fromarray(np.uint8(cm.gist_earth(horizontal)*255))
smoothed = converted_img.filter(ImageFilter.ModeFilter(size=13))
#smoothed = np.array(smoothed)
printBeforeAndAfter('Smoothed', horizontal, smoothed)
#%%
# Find furthest point
# Draw closest point
blank_image = np.zeros((h,w,3), np.uint8)
draw = ImageDraw.Draw(smoothed)
draw.point((150,220), 'red')

# Find furthest point
thres, bw_img = cv2.threshold(np.asanyarray(smoothed),127,255,cv2.THRESH_BINARY)
print(type(bw_img))
print(bw_img[150,220])
print(bw_img[221,150])
bw_img[150,220] = [0,255,0,255]
    

#for i in bw_img:
#    for j in i:
#        if j[-1] == 255:
#            j[3] = 0
#print(first_white_pixel)
    #draw.point(i, 'red')

printBeforeAndAfter('Points', smoothed, bw_img)
