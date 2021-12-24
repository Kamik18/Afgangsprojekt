import cv2
import os
import voronoi_road_map
import numpy as np
import matplotlib.pyplot as plt

#ori=cv2.imread('ICP/Kort.png')
ori=cv2.imread('ICP/Kort.png')

medianBlur = cv2.medianBlur(ori, 9)
median_edges = cv2.Canny(medianBlur, 50, 130)

rows, cols = ori.shape[:2]
print(cols, rows)

onlyEdges_x = []
onlyEdges_y = []

for i in range(1, rows -1):
    for j in range(1, cols -1):
        if median_edges[i,j] == 255:
            onlyEdges_x.append(i)
            onlyEdges_y.append(j)
            
empty = np.zeros(ori.shape[:2])

for i in range(len(onlyEdges_x)):
    empty[onlyEdges_x[i],onlyEdges_y[i]] = 255


#cv2.imshow('original',ori)
#cv2.imshow('blur',medianBlur)
#cv2.imshow('edges',median_edges)
#cv2.imshow('reinsert',empty)
#cv2.waitKey(0)


from skimage.morphology import medial_axis, skeletonize, thin

image = cv2.cvtColor(ori, cv2.COLOR_RGB2GRAY)
for i in range(1, rows -1):
    for j in range(1, cols -1):
        if image[i,j] == 255:
            image[i,j] = 1

skeleton = skeletonize(image)
thinned = thin(image)
thinned_partial = thin(image, max_num_iter=25)
skel, distance = medial_axis(image, return_distance=True)
dist_on_skel = distance * skel
fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
ax = axes.ravel()

# Convert from dtype: bool to uint8
skeleton_uint8 = skeleton.astype(np.uint8)

for i in range(1, rows -1):
    for j in range(1, cols -1):
        if skeleton_uint8[i,j] == True:
            skeleton_uint8[i,j] = 255
        else:
            skeleton_uint8[i,j] = 0
        if image[i,j] == 1:
            image[i,j] = 255

skeleton_color = cv2.cvtColor(skeleton_uint8, cv2.COLOR_GRAY2RGB) 
image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
dots = np.zeros(image_color.shape)
for i in range(1, rows -1):
    for j in range(1, cols -1):
        if skeleton_color[i,j][0] == 255:
            skeleton_color[i,j] = [255,0,0]
            image_color[i,j] = [255,0,0]
        counter = 0
        for k in range(-1, 2, 1):
            for l in range(-1, 2, 1):
               if skeleton_uint8[i+l,j+k] == 255: 
                   counter += 1
        if counter > 3:
            cv2.circle(image_color, (j,i), 1, (0,255,0), 1)

cv2.imshow('end result', image_color)
cv2.resizeWindow('end result', int(len(image_color[0]*2)), int(len(image_color[1])*2))
cv2.waitKey(0)

ax[0].imshow(image_color, cmap=plt.cm.gray)
ax[0].set_title('original')
ax[0].axis('off')

ax[1].imshow(skeleton_color, cmap=plt.cm.gray)
ax[1].set_title('skeleton')
ax[1].axis('off')

ax[2].imshow(dist_on_skel, cmap=plt.cm.gray)
ax[2].set_title('thinned')
ax[2].axis('off')

ax[3].imshow(thinned_partial, cmap=plt.cm.gray)
ax[3].set_title('partially thinned')
ax[3].axis('off')



fig.tight_layout()
plt.show()