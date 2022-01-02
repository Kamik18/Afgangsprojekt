#%%
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

start = time.time()
# # Edge detection - Canny
#img_dir = '../data/AlfaLavalFinal/images'
img_dir = './images'
num = '300'

img = cv2.imread(f'{img_dir}/Alfa_Laval_Sensor_{num}.jpg')
#img = cv2.imread('obstacle_scene_1.jpg')

# Convert image color and blur image
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gaussianBlur = cv2.GaussianBlur(img, (3,3),cv2.BORDER_DEFAULT)
bilateralBlur = cv2.bilateralFilter(img, 9,40,75)
medianBlur = cv2.medianBlur(img, 5)
# Sobel Edge Detection
img_gray_blur = cv2.cvtColor(medianBlur, cv2.COLOR_RGB2GRAY)

ret, thresh_gray = cv2.threshold(img_gray_blur, 85, 255, cv2.THRESH_BINARY)

sobelxy = cv2.Sobel(src=img_gray_blur, ddepth=cv2.CV_8U, dx=1, dy=1, ksize=5, borderType=cv2.BORDER_DEFAULT) # Combined X and Y Sobel Edge Detection
laplacian = cv2.Laplacian(thresh_gray,cv2.CV_8U, ksize=5, borderType=cv2.BORDER_DEFAULT)
#laplacian = cv2.inRange(laplacian, 50, 255)
# Edge detection
edges = cv2.Canny(img, 50,130)
gaussian_edges = cv2.Canny(gaussianBlur, 50, 130)
bilateral_edges = cv2.Canny(bilateralBlur, 50, 130)
median_edges = cv2.Canny(medianBlur, 50, 130)

# end time
end = time.time()
print(f"Runtime of the program is {end - start}")
# Print
printBeforeAndAfter('Original', 'No Blur Edges', img, edges)
printBeforeAndAfter('No Blur', 'Gaussian Blur Edges', edges, gaussian_edges)
printBeforeAndAfter('Bilateral Blur Edges', 'Median Blur Edges ',img, median_edges)
printBeforeAndAfter('Sobel Gaussian Edges', 'Laplacian Gaussian edges',thresh_gray, laplacian)
"""
cv2.imwrite(f'./test/Alfa_Laval_Sensor_Edges_{num}.jpg', edges)
cv2.imwrite(f'./test/Alfa_Laval_Sensor_Gaussian_Canny_Edges_{num}.jpg', gaussian_edges)
cv2.imwrite(f'./test/Alfa_Laval_Sensor_Median_Canny_Edges_{num}.jpg', median_edges)
cv2.imwrite(f'./test/Alfa_Laval_Sensor_Bilateral_Canny_Edges_{num}.jpg', bilateral_edges)

cv2.imwrite(f'./test/Alfa_Laval_Sensor_Sobel_Edges_{num}.jpg', sobelxy)
"""

#%%
# Sidefill
#gaussian_edges = median_edges
#edges = bilateral_edges
start = time.time()
#edges = blurred_edge2
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
row_inds_at_edges_blur[gaussian_edges==0] = 0 # only get indices at edges, 0 elsewhere
max_row_inds_blur = np.amax(row_inds_at_edges_blur, axis=0) # find the max row ind over each col

inds_after_edges_blur = row_inds >= max_row_inds_blur

filled_from_bottom_blur = np.zeros((h, w))
filled_from_bottom_blur[inds_after_edges_blur] = 255

# end time
end = time.time()
print(f"Runtime of the program is {end - start}")
printBeforeAndAfter('Sidefill', 'Sidefill Blur', filled_from_bottom, filled_from_bottom_blur)
#%%
# Horizontal dialate and erode
start = time.time()
# Copy images
horizontal = filled_from_bottom.copy()
horizontal_blur = filled_from_bottom_blur.copy()

# Horizontal erode
horizontal_size_erode = w // 30
horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size_erode, 1))
horizontal = cv2.erode(horizontal, horizontalStructure)
horizontal_blur = cv2.erode(horizontal_blur, horizontalStructure)

# Horizontal dilate
# This is done to remove small corns found in the floor
horizontal_size_dialate = w // 20
horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size_dialate, 1))
horizontal = cv2.dilate(horizontal, horizontalStructure)
horizontal_blur = cv2.dilate(horizontal_blur, horizontalStructure)
# Horizontal erode
horizontal_size_erode = w // 30
horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size_erode, 1))
horizontal = cv2.erode(horizontal, horizontalStructure)
horizontal_blur = cv2.erode(horizontal_blur, horizontalStructure)

# end time
end = time.time()
print(f"Runtime of the program is {end - start}")
printBeforeAndAfter('Horizontal Erode', 'Horizontal Erode Blur', horizontal, horizontal_blur)
#%%
# Smoothed
start = time.time()
converted_img = Image.fromarray(np.uint8(cm.gist_earth(horizontal)*255))
smoothed = converted_img.filter(ImageFilter.ModeFilter(size=13))

converted_img_blur = Image.fromarray(np.uint8(cm.gist_earth(horizontal_blur)*255))
smoothed_blur = converted_img_blur.filter(ImageFilter.ModeFilter(size=13))


#smoothed_blur = horizontal_blur #= cv2.dilate(horizontal_blur, kernel=kernel1)
#smoothed_blur = horizontal_blur = cv2.erode(horizontal_blur, kernel=kernel)
# end time
end = time.time()
print(f"Runtime of the program is {end - start}")
printBeforeAndAfter('Smoothed', 'Smoothed Blur', smoothed, smoothed_blur)
#%%
# Find furthest point
start = time.time()
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

# Find furthest point 
img_with_alpha_values = np.asarray(smoothed)
img_with_alpha_values_blur = np.asarray(smoothed_blur)
# Set robot center point
#createDot(img_with_alpha_values, (int(h*0.9), int(w/2)))
#createDot(img_with_alpha_values_blur, (int(h*0.9), int(w/2)))

# Convert image to grayscale and threshold it
bw = cv2.cvtColor(img_with_alpha_values, cv2.COLOR_RGB2GRAY)
bw = cv2.inRange(bw, 190, 255)

bw_blur = cv2.cvtColor(img_with_alpha_values_blur, cv2.COLOR_RGB2GRAY)
bw_blur = cv2.inRange(bw_blur, 190, 255)

# Find all occurences of white pixels
print(w, h)
pixel_y, pixel_x = np.nonzero(bw)
createDot(img_with_alpha_values, (pixel_y[0], pixel_x[0]))

pixel_y_blur, pixel_x_blur = np.nonzero(bw_blur)
createDot(img_with_alpha_values_blur, (pixel_y_blur[0], pixel_x_blur[0]))

# end time
end = time.time()
print(f"Runtime of the program is {end - start}")
# Print
printBeforeAndAfter('Points', 'Points Blur', img_with_alpha_values, img_with_alpha_values_blur)

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
# Final
# Construct a colour image to superimpose
start = time.time()
mask = np.asarray(img_with_alpha_values)
mask_blur = np.asarray(img_with_alpha_values_blur)

img_with_overlay = cv2.addWeighted(img, 0.9, mask[:,:,:3], 0.3, 0)
img_with_overlay_blur = cv2.addWeighted(gaussianBlur, 0.9, mask_blur[:,:,:3], 0.3, 0)
createDot(img_with_overlay, (pixel_y[0], pixel_x[0]))
createDot(img_with_overlay_blur, (pixel_y_blur[0], pixel_x_blur[0]))
# end time
end = time.time()
print(f"Runtime of the program is {end - start}")
#printBeforeAndAfter('final', img_masked, img_masked_blur)
printBeforeAndAfter('Final', 'Final Blur', img_with_overlay, img_with_overlay_blur)
#cv2.imwrite(f'./test/Alfa_Laval_Sensor_Test_Bilateral_{num}.jpg', img_with_overlay)
#cv2.imwrite(f'./test/Alfa_Laval_Sensor_Test_Median_{num}.jpg', img_with_overlay_blur)
#cv2.imwrite(f'./test/Alfa_Laval_Sensor_Test_{num}.jpg', img_with_overlay)
#cv2.imwrite(f'./test/Alfa_Laval_Sensor_Test_Gaussian_{num}.jpg', img_with_overlay_blur)

# %%
print(pixel_y_blur[0])
print(pixel_x_blur[0])
#%%
x = pixel_x_blur[0]
y = pixel_y_blur[0]
imcopy = img_with_overlay_blur.copy()
# calculate width
dist = 400
# Convert 21850 * x^-1.033 to non-negative power
width = 21850 * (1/(dist**1.033))
print('width = ', width)
left = (int(x - (width/2)), y)

print(left)
right = (int(x + (width/2)), y)
print(right)
# Add line
cv2.circle(imcopy, (x, y), 2, (255,0,0), 2)
cv2.circle(imcopy, (x, y), 2, (255,0,0), 2)
cv2.line(imcopy, left, right, (0,0,255), 2)
fig = plt.figure(figsize=(15,10))
fig.add_subplot(1,1, 1)
plt.imshow(imcopy)
