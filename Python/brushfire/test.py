#%%
from PIL.Image import new
from numpy.linalg import norm 
from scipy.spatial import distance
import numpy as np
from dataclasses import dataclass

@dataclass
class Point:
    x: int
    y: int

current = Point(263, 93)
oi = Point(263, 274)
x_dist = np.sqrt(abs(current.x - oi.x)) ** 2
y_dist = np.sqrt(abs(current.y - oi.y)) ** 2
new_dist = x_dist + y_dist


a = [263, 93]
b = [263, 274]

dist = np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
#dist = distance.euclidean(a,b)
#dist = norm(a - b)
print('new_dist: ', new_dist)
print(dist)
#%%
import cv2
import os


img = cv2.imread('../ICP/Kort.png', cv2.IMREAD_GRAYSCALE)
cols, rows = img.shape[:2]
print(img.shape)
#cv2.imshow('s', img)
#cv2.waitKey(0 )