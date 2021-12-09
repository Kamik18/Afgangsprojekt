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
    fig.subplots_adjust(top=1.5)
    rows = 1
    columns = 2
    fig.add_subplot(rows,columns, 1)
    plt.title('original')
    plt.imshow(before)
    fig.add_subplot(rows,columns, 2)
    plt.title('Blurred')
    plt.imshow(after)
    
#%%
# # Edge detection - Canny
img_dir = '../data/Background'
num = '128'

img = cv2.imread(f'{img_dir}/Alfa_Laval_Sensor_{num}.jpg')
#img = cv2.imread('obstacle_scene_1.jpg')

# Convert image color and blur image
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_blur = cv2.GaussianBlur(img, (3,3),cv2.BORDER_DEFAULT)

# Sobel Edge Detection
img_gray_blur = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
sobelxy = cv2.Sobel(src=img_gray_blur, ddepth=cv2.CV_8U, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
laplacian = cv2.Laplacian(img_gray_blur,cv2.CV_8U, ksize=3, borderType=cv2.BORDER_DEFAULT)
#laplacian = cv2.inRange(laplacian, 20, 255)
# Edge detection
edges = cv2.Canny(img, 100,200)
blurred_edge = cv2.Canny(img_blur, 100,200)

# Print
printBeforeAndAfter('Images', img, img_blur)
printBeforeAndAfter('Canny', edges, blurred_edge)
printBeforeAndAfter('Sobel', laplacian, sobelxy)

#%%
# Sidefill
blurred_edge = laplacian
h, w = img.shape[:2]
row_inds = np.indices((h, w))[0] # gives row indices in shape of img
row_inds_at_edges = row_inds.copy()
row_inds_at_edges[edges==0] = 0 # only get indices at edges, 0 elsewhere
max_row_inds = np.amax(row_inds_at_edges, axis=0) # find the max row ind over each col

inds_after_edges = row_inds >= max_row_inds

filled_from_bottom = np.zeros((h, w))
filled_from_bottom[inds_after_edges] = 255

# Blurred image sidefill
row_inds_at_edges_blur = row_inds.copy()
row_inds_at_edges_blur[blurred_edge==0] = 0 # only get indices at edges, 0 elsewhere
max_row_inds_blur = np.amax(row_inds_at_edges_blur, axis=0) # find the max row ind over each col

inds_after_edges_blur = row_inds >= max_row_inds_blur

filled_from_bottom_blur = np.zeros((h, w))
filled_from_bottom_blur[inds_after_edges_blur] = 255

printBeforeAndAfter('Sidefill', filled_from_bottom, filled_from_bottom_blur)
#%%
# Horizontal erode
horizontal = filled_from_bottom
#horizontal_size = w // 6
horizontal_size = w // 30
horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
horizontal = cv2.erode(horizontal, horizontalStructure)

# Horizontal erode
horizontal_blur = filled_from_bottom_blur
horizontal_blur = cv2.erode(horizontal_blur, horizontalStructure)

printBeforeAndAfter('horizontal erode', horizontal, horizontal_blur)
#%%
# Smoothed
converted_img = Image.fromarray(np.uint8(cm.gist_earth(horizontal)*255))
smoothed = converted_img.filter(ImageFilter.ModeFilter(size=13))

converted_img_blur = Image.fromarray(np.uint8(cm.gist_earth(horizontal_blur)*255))
smoothed_blur = converted_img_blur.filter(ImageFilter.ModeFilter(size=13))

printBeforeAndAfter('Smoothed', smoothed, smoothed_blur)
#%%
# Find furthest point
# Draw closest point
def createDot(img, point):
    for i in range(-5, 6, 1):
        for j in range(-5, 6, 1):
            x = point[0] + i
            y = point[1] + j
            img[x, y] = [255,0,0,255]    

# Find furthest point 
img_with_alpha_values = np.asarray(smoothed)
img_with_alpha_values_blur = np.asarray(smoothed_blur)

# Set robot center point
createDot(img_with_alpha_values, (int(w/2), int(h*0.9)))
createDot(img_with_alpha_values_blur, (int(w/2), int(h*0.9)))

# Convert image to grayscale and threshold it
bw = cv2.cvtColor(img_with_alpha_values, cv2.COLOR_RGB2GRAY)
bw = cv2.inRange(bw, 190, 255)

bw_blur = cv2.cvtColor(img_with_alpha_values_blur, cv2.COLOR_RGB2GRAY)
bw_blur = cv2.inRange(bw_blur, 190, 255)

# Find all occurences of white pixels
pixel_y, pixel_x = np.nonzero(bw)
print('pixel_x {}, pixel_y {}'.format(pixel_x[0], pixel_y[0]))
createDot(img_with_alpha_values, (pixel_y[0], pixel_x[0]))

pixel_y_blur, pixel_x_blur = np.nonzero(bw_blur)
print('pixel_x {}, pixel_y {}'.format(pixel_y_blur[0], pixel_x_blur[0]))
createDot(img_with_alpha_values_blur, (pixel_y_blur[0], pixel_x_blur[0]))

# Print
printBeforeAndAfter('Points', img_with_alpha_values, img_with_alpha_values_blur)

# Test to select first large white row
"""
counter = 0
iterateCounter = 0
for i in range(pixel_x):
    if pixel_x[i+1] - 1 == pixel_x[i]:
        iterateCounter +=1
    else:
        placedot = i - 1
        counter = iterateCounter
        iterateCounter = 0
    
    if iterateCounter > 30:

        createDot(bw_img, (pixel_y[i]))
"""




#%%
# Construct a colour image to superimpose
mask = np.asarray(img_with_alpha_values)
mask_blur = np.asarray(img_with_alpha_values_blur)

# Convert the input image and color mask to Hue Saturation Value (HSV) colorspace
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
smoothed_hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)

img_hsv_blur = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
smoothed_hsv_blur = cv2.cvtColor(mask_blur, cv2.COLOR_BGR2HSV)

# Create bitwise_or to add the two images together
img_hsv_mask = cv2.bitwise_or(img_hsv, smoothed_hsv)
img_masked = cv2.cvtColor(img_hsv_mask, cv2.COLOR_HSV2BGR)

img_hsv_mask_blur = cv2.bitwise_or(img_hsv_blur, smoothed_hsv_blur)
img_masked_blur = cv2.cvtColor(img_hsv_mask_blur, cv2.COLOR_HSV2BGR)

fig = plt.figure(figsize=(20,15))
fig.suptitle('overlay', fontsize=16)
fig.subplots_adjust(top=1.5)
rows = 1
columns = 3
fig.add_subplot(rows,columns, 1)
plt.title('original')
plt.imshow(img)
fig.add_subplot(rows,columns, 2)
plt.title('Overlay')
plt.imshow(mask)
fig.add_subplot(rows,columns, 3)
plt.title('combined')
plt.imshow(img_masked)

fig = plt.figure(figsize=(20,15))
fig.suptitle('overlay', fontsize=16)
fig.subplots_adjust(top=1.5)
rows = 1
columns = 3
fig.add_subplot(rows,columns, 1)
plt.title('original')
plt.imshow(img_blur)
fig.add_subplot(rows,columns, 2)
plt.title('Overlay')
plt.imshow(mask_blur)
fig.add_subplot(rows,columns, 3)
plt.title('combined')
plt.imshow(img_masked_blur)