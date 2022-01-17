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
#C:/Usersemila/Documents/Github/Afgangsprojekt/Python/ICP/Kort.png
#brushfire_img = brushfire_metohds.BrushfireAlgorithmGrayScale('ICP/Corrected_optimized_colored.png', 3)
brushfire_img = brushfire_metohds.BrushfireAlgorithmGrayScale('C:/Users/emila/Documents/Github/Afgangsprojekt/Python/ICP/Corrected_optimized_colored.png', 3)
#brushfire_img = brushfire_metohds.BrushfireAlgorithmGrayScale('ICP/Kort.png', 3)

ori=cv2.imread('C:/Users/emila/Documents/Github/Afgangsprojekt/Python/ICP/Kort.png')
print(ori.shape)
exit(0)
ori = cv2.resize(ori, (1032, 785))
#cv2.imwrite('brushfire/Kort.png', ori)
workspacemap = np.zeros((1032, 785), np.uint8)

cols, rows = ori.shape[:2]
cols = cols - 1
rows = rows - 1
for i in range(1,cols):
    for j in range(1, rows): 
        if ori[i,j] < 34:
            ori[i,j] = 0
ori = cv2.cvtColor(ori, cv2.COLOR_GRAY2RGB)
# Add workspace areas
workspace_img = brushfire_metohds.workspace(brushfire_img, ori, 10)

for i in range(1,cols):
    for j in range(1, rows): 
        if ori[i,j] == [255, 0, 0]:
            workspacemap[i,j] = 255

cv2.imshow('brushfire', brushfire_img)
cv2.imshow('workspace', workspace_img)
cv2.imshow('workspacemap', workspacemap)
#cv2.imwrite('brushfire/brushfire.png', brushfire_img)
#cv2.imwrite('brushfire/workspace.png', workspace_img)
#cv2.imwrite('brushfire/original.png', ori)
cv2.waitKey(0)
