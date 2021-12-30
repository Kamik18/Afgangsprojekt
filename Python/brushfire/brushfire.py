from matplotlib.pyplot import imshow
from brushfire_metohds import workspace
import brushfire_metohds
import scanner
import cv2
import os
import numpy as np
from numpy.linalg import norm 
import math
from dataclasses import dataclass

brushfire_img = brushfire_metohds.BrushfireAlgorithmGrayScale('ICP/Kort.png', 3)
ori=cv2.imread('ICP/Kort.png')


# Add workspace areas
workspace_img = brushfire_metohds.workspace(brushfire_img, ori, 10)

cv2.imshow('brushfire', brushfire_img)
cv2.imshow('ori', workspace_img)
cv2.imwrite('ICP/brushfire.png', brushfire_img)
cv2.waitKey(0)
exit(1)

#reduced_img, point = brushfire_metohds.ReduceToLowestResolution(brushfire_img)

#img = brushfire_img
img=cv2.imread('ICP/Kort.png', cv2.IMREAD_GRAYSCALE)
img = cv2.medianBlur(img, 9)
# Method starts here
rows, cols = img.shape[:2]
cols = cols - 1
rows = rows - 1

color_img = img.copy()
color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

@dataclass
class Point:
    x: int
    y: int

"""
# TODO: Append to array and do an 'any()?' check to make sure white pixels are present.
# Or just break if a white pixel is detected?
p1 = np.array([137,20])
p2 = np.array([138,25]) # Two example points.
#p1 = Point(137,20)
#p2 = Point(138,25)
p = p1; print('p: ', p[0])
d = p2-p1; print('d: ', d)
N = np.max(np.abs(d)); print('N: ', N)
s = d/N; print('s: ', s)
print(np.rint(p).astype('int'))
for ii in range(0,N):
   p = p+s
   print(np.rint(p).astype('int'), ' val: ', img[int(p[0]), int(p[1])])
   print()
   if img[int(p[0]), int(p[1])] == 255:
       print("The two points are different obstacles")
       break
    

exit(0)
"""

for j in range(1, rows - 1): 
    for i in range(1, cols - 1):
        scanner.getSensorData(color_img, img, Point(i, j), 80, 8)

cv2.imshow("tangentBug", color_img)
cv2.waitKey(0)

#cv2.imshow('Voronoi', color_img)
#cv2.imshow('Reduced brushfire', reduced_img)
#cv2.waitKey()

"""
def CheckKernel(point):
    x = point[0]
    y = point[1]
    
    maxes = []
    points = []
    max_object = {}
    center_val = img[x,y]
    print('center_val: ', center_val)
    # Get all values around the pixels in a 3x3 kernel and add to lists
    for i in range(-1, 2, 1):
        for j in range(-1, 2, 1):
            print('at point: (' , x + j, ', ', y + i, ') - val: ', img[x + j,y + i])
            if (x + j, y + i) in visited_points:
                #print('found ', (y + i, x + j))
                continue
            if (x + j, y + i) == point:
                continue
            if img[x + j, y + i] == center_val - 1:
                maxes.append(img[x + j, y + i])
                points.append((x + j, y + i))
    
    # Find max value in the list
    
    #max_value = max(maxes)
    #print('max_value: ', max_value)

    # Create new list containing only highest values
    #new_maxes = []
    #new_points = []
    #for i in range(len(maxes)):
        #print('point: ', points[i], 'value: ', maxes[i])
        #if maxes[i] == max_value:
            #if points[i] == point: # Value four is the one we are standing at
            #    continue
        #    new_maxes.append(maxes[i])
        #    new_points.append(points[i])

    #print('new points', new_points)
    return maxes, points

# Start from the max brightness.
# For each point around the middle, that is only 1 value lower, add a new path to the queue that goes that way
# Once a path reaches 0, kill it.
#cv2.circle(color_img, (150, 150), 2, (0,0,255),2)
#maxes = CheckKernel(201, 227)
visited_points = set([])
points_to_check = set([(point[0], point[1])])
#points_to_check = set([(150,150)])
step = 0
while step < 1:
    print('iteration nr: ', step)
    print('before ', points_to_check)
    # Get first point in list
    point = points_to_check.pop()
    #print('withdraw ', point)
    print('after ', points_to_check)
    # Check if point exists in points_to_check
    print('point: ', point)
    #print('visited points: ', visited_points)
    #if point in visited_points:
    #    print('found ', point)
    #    continue
    # If point doesn't exist, check surrounding points
    maxes, points = CheckKernel(point)
    if not maxes[0] == img(point):
        # Add new point to visited
        visited_points.add(point)
    # Color visited point
    color_img[point] = [0,0,255]
    # Add new points to points_to_check
    points_to_check.update(points) 
    #print('points to check: ', points_to_check)
    #print('------------------------')
    step += 1
"""



    
