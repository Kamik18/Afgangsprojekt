import cv2
import os
#import voronoi_road_map
import numpy as np
import matplotlib.pyplot as plt

#ori=cv2.imread('ICP/Kort.png')
ori=cv2.imread('ICP/Kort.png')

medianBlur = cv2.medianBlur(ori, 9)
median_edges = cv2.Canny(medianBlur, 50, 130)

rows, cols = ori.shape[:2]

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

# Image needs to be 0 and 1 in order to work with skeletonize
image = cv2.cvtColor(ori, cv2.COLOR_RGB2GRAY)
for i in range(1, rows -1):
    for j in range(1, cols -1):
        if image[i,j] == 255:
            image[i,j] = 1

skeleton = skeletonize(image)
skeleton_lee = skeletonize(image, method='lee')
thinned = thin(image)
thinned_partial = thin(image, max_num_iter=25)

skel, distance = medial_axis(image, return_distance=True)
dist_on_skel = distance * skel

print(skel)
# Convert from dtype: bool to uint8
skeleton_uint8 = skeleton.astype(np.uint8)
skel_uint8 = skel.astype(np.uint8)
skeleton_lee_color = cv2.cvtColor(skeleton_lee, cv2.COLOR_GRAY2RGB)
print(skel_uint8)
for i in range(1, rows -1):
    for j in range(1, cols -1):
        # Revert skeletonize to 255 and 0
        if skeleton_uint8[i,j] == True:
            skeleton_uint8[i,j] = 255
        else:
            skeleton_uint8[i,j] = 0
        # convert image to 255 and 0
        if image[i,j] == 1:
            image[i,j] = 255
            
        if skel_uint8[i,j] > 0:
            skel_uint8[i,j] = 255

        if skeleton_lee[i,j] > 0:
            skeleton_lee[i,j] = 255
            skeleton_lee_color[i,j] = [255,255,255]
        

# Convert to RGB
skeleton_color = cv2.cvtColor(skeleton_uint8, cv2.COLOR_GRAY2RGB) 
image_color_skeleton = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
#dist_on_skel_uint8 = cv2.cvtColor(dist_on_skel_uint8, cv2.COLOR_GRAY2RGB)
skeleton_lee = cv2.cvtColor(skeleton_lee, cv2.COLOR_GRAY2RGB)
image_color_medialAxis = image_color_skeleton.copy()

for i in range(1, rows -1):
    for j in range(1, cols -1):
        # Insert red lines where skeleton fits the original image
        if skeleton_color[i,j][0] == 255:
            #skeleton_color[i,j] = [255,0,0]
            image_color_skeleton[i,j] = [255,0,0]
        # Create dots
        counter_skel = 0
        for k in range(-1, 2, 1):
            for l in range(-1, 2, 1):
               if skeleton_uint8[i+l,j+k] == 255: 
                   counter_skel += 1
        if counter_skel > 3:
            cv2.circle(image_color_skeleton, (j,i), 2, (0,255,0), 2)

        if dist_on_skel[i,j] > 0.0:
            image_color_medialAxis[i,j] = [255,0,0]
        counter_dist = 0
        for k in range(-1, 2, 1):
            for l in range(-1, 2, 1):
               if dist_on_skel[i+l,j+k] > 0: 
                   counter_dist += 1
        if counter_dist > 3:
            cv2.circle(image_color_medialAxis, (j,i), 2, (0,255,0), 2)

fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(ori, cmap=plt.cm.gray)
ax[0].set_title('original')
ax[0].axis('off')

ax[1].imshow(skeleton_color, cmap=plt.cm.gray) 
ax[1].set_title('skeleton')
ax[1].axis('off')

ax[2].imshow(dist_on_skel, cmap=plt.cm.gray)
ax[2].set_title('Medial Axis')
ax[2].axis('off')

ax[3].imshow(image_color_skeleton, cmap=plt.cm.gray)
ax[3].set_title('Image with Skeleton')
ax[3].axis('off')

ax[4].imshow(skeleton_lee, cmap=plt.cm.gray)
ax[4].set_title('skeleton lee')
ax[4].axis('off')

ax[5].imshow(image_color_medialAxis, cmap=plt.cm.gray)
ax[5].set_title('Image with Medial axis')
ax[5].axis('off')


#cv2.imwrite('voronoi/skeleton.png', skeleton_color)
#cv2.imwrite('voronoi/skeleton_map.png', image_color_skeleton)
#cv2.imwrite('voronoi/medialAxis_map.png', image_color_medialAxis)
cv2.imwrite('voronoi/medialAxis.png', skel_uint8)
#cv2.imwrite('voronoi/skeleton_lee.png', skeleton_lee)

fig.tight_layout()
plt.show()


#cv2.imshow('end result median axis', image_color_medialAxis)
#cv2.imwrite('voronoi/skeleton.png', image_color_skeleton)
#cv2.imshow('end result skel', image_color_skeleton)
#cv2.imwrite('voronoi/medialaxis.png', image_color_medialAxis)
#cv2.waitKey(0)