import cv2
import os
#import voronoi_road_map
import numpy as np
import matplotlib.pyplot as plt

#ori=cv2.imread('ICP/Kort.png')
#ori=cv2.imread('ICP/MapNoPoints.png')
#print(os.listdir())
#ori=cv2.imread('ICP/Corrected_optimized_colored.png')
#ori=cv2.imread('C:/Users\emila/Documents/Github/Afgangsprojekt/Python/ICP/Corrected_optimized_colored.png')
ori=cv2.imread('C:/Users\emila/Documents/Github/Afgangsprojekt/Python/ICP/Kort.png')


medianBlur = cv2.medianBlur(ori, 9)
median_edges = cv2.Canny(medianBlur, 50, 130)

#cv2.imshow('s', median_edges)
#cv2.waitKey(0)
#exit(1)


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
#image = cv2.cvtColor(ori, cv2.COLOR_RGB2GRAY)

gray = cv2.cvtColor(ori, cv2.COLOR_RGB2GRAY)
thresh, image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
kernel = np.ones((5, 5), np.uint8)
image = cv2.erode(image, kernel)
image = cv2.dilate(image, kernel)



#cv2.imshow('s', image)
#cv2.waitKey(0)
#exit(1)
#print(image)
for i in range(0, rows):
    for j in range(0, cols):
        if image[i,j] == 255:
            image[i,j] = 1
        else: 
            image[i,j] = 0

skeleton = skeletonize(image)
skeleton_lee = skeletonize(image, method='lee')
#skel, distance = medial_axis(image, return_distance=True)
skel = medial_axis(image, return_distance=False)

# Convert from dtype: bool to uint8
skeleton_uint8 = skeleton.astype(np.uint8)
skel_uint8 = skel.astype(np.uint8)
#skeleton_lee_color = cv2.cvtColor(skeleton_lee, cv2.COLOR_GRAY2RGB)

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
            #skeleton_lee_color[i,j] = [255,255,255]
        

# Convert to RGB
#skeleton_color = cv2.cvtColor(skeleton_uint8, cv2.COLOR_GRAY2RGB) 
image_color_skeleton = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
image_color_skeleton_lee = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
#image_color_medialAxis = image_color_skeleton.copy()
image_color_medialAxis =  cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

for i in range(1, rows -1):
    for j in range(1, cols -1):
        # Insert red lines where skeleton fits the original image
        if skeleton[i,j] > 0:
            image_color_skeleton[i,j] = [255,0,0]
            # Create dots
            counter_skel = 0
            for k in range(-1, 2, 1):
                for l in range(-1, 2, 1):
                    if l == 0 and k == 0:
                        continue
                    if skeleton_uint8[i+l,j+k] == 255: 
                        counter_skel += 1
            if counter_skel > 2:
                #image_color_skeleton[i,j] = [0,0,255]
                cv2.circle(image_color_skeleton, (j,i), 1, (0,0,255), 1)

        if skeleton_lee[i,j] > 0:
            image_color_skeleton_lee[i,j] = [255,0,0]
            counter_dist = 0
            for k in range(-1, 2, 1):
                for l in range(-1, 2, 1):
                    if l == 0 and k == 0:
                        continue
                    if skeleton_lee[i+l,j+k] > 0: 
                        counter_dist += 1
            if counter_dist > 2:
                #image_color_skeleton_lee[i,j] = [0,0,255]
                cv2.circle(image_color_skeleton_lee, (j,i), 1, (0,0,255), 1)

        if skel_uint8[i,j] > 0:
            image_color_medialAxis[i,j] = [255,0,0]
            counter_dist = 0
            for k in range(-1, 2, 1):
                for l in range(-1, 2, 1):
                    if l == 0 and k == 0:
                        continue
                    if skel_uint8[i+l,j+k] > 0: 
                        counter_dist += 1
            if counter_dist > 2:
                #image_color_medialAxis[i,j] = [0,0,255]
                cv2.circle(image_color_medialAxis, (j,i), 1, (0,0,255), 1)

fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(skeleton, cmap=plt.cm.gray)
ax[0].set_title('skeleton')
ax[0].axis('off')

ax[1].imshow(skeleton_lee, cmap=plt.cm.gray) 
ax[1].set_title('skeleton lee')
ax[1].axis('off')

ax[2].imshow(skel, cmap=plt.cm.gray)
ax[2].set_title('Medial Axis')
ax[2].axis('off')

ax[3].imshow(image_color_skeleton, cmap=plt.cm.gray)
ax[3].set_title('Image with Skeleton')
ax[3].axis('off')

ax[4].imshow(image_color_skeleton_lee, cmap=plt.cm.gray)
ax[4].set_title('Image with Skeleton Lee')
ax[4].axis('off')

ax[5].imshow(image_color_medialAxis, cmap=plt.cm.gray)
ax[5].set_title('Image with Medial axis')
ax[5].axis('off')

"""
# Skeleton
cv2.imwrite('voronoi/icp_skeleton.png', skeleton_uint8)
cv2.imwrite('voronoi/icp_skeleton_map.png', image_color_skeleton)
# Skeleton lee
cv2.imwrite('voronoi/icp_skeleton_lee.png', skeleton_lee)
cv2.imwrite('voronoi/icp_skeleton_lee_map.png', image_color_skeleton_lee)
# Medial
cv2.imwrite('voronoi/icp_medialAxis.png', skel_uint8)
cv2.imwrite('voronoi/icp_medialAxis_map.png', image_color_medialAxis)
"""

fig.tight_layout()
plt.show()


#cv2.imshow('end result median axis', image_color_medialAxis)
#cv2.imwrite('voronoi/skeleton.png', image_color_skeleton)
#cv2.imshow('end result skel', image_color_skeleton)
#cv2.imwrite('voronoi/medialaxis.png', image_color_medialAxis)
#cv2.waitKey(0)