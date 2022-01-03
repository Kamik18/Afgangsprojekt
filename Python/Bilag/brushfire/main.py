import brushfire
import cv2

brushfire_img = brushfire.BrushfireAlgorithmGrayScale('ICP/Kort.png', 3)
ori=cv2.imread('ICP/Kort.png')


# Add workspace areas
workspace_img = brushfire.workspace(brushfire_img, ori, 10)

cv2.imshow('brushfire', brushfire_img)
cv2.imshow('ori', workspace_img)
cv2.imwrite('ICP/brushfire.png', brushfire_img)
cv2.waitKey(0)
