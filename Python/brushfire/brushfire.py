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
