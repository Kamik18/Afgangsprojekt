import cv2 as cv
import numpy as np

img_rgb = cv.imread('map.jpg')
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
template = cv.imread('Position_1.png',0)
cv.imshow('template 1', template)

scale_percent = 24
template = cv.resize(template, (int(template.shape[1] * scale_percent / 100), int(template.shape[0] * scale_percent / 100)), interpolation = cv.INTER_AREA)
w, h = template.shape[::-1]
res = cv.matchTemplate(img_gray,template, cv.TM_CCOEFF_NORMED)
threshold = 0.25
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv.circle(img_rgb, (pt[0] + int(w / 2), pt[1] + int(h / 2)), 50, (0,0,255), 2)
cv.imshow('Res 1', img_rgb)
cv.waitKey(0)
cv.destroyAllWindows()

img_rgb = cv.imread('map.jpg')
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
template = cv.imread('Position_2 - Copy.png',0)
cv.imshow('template 2', template)

scale_percent = 24
template = cv.resize(template, (int(template.shape[1] * scale_percent / 100), int(template.shape[0] * scale_percent / 100)), interpolation = cv.INTER_AREA)
w, h = template.shape[::-1]
res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
threshold = 0.28
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv.circle(img_rgb, (pt[0] + int(w / 2), pt[1] + int(h / 2)), 50, (0,0,255), 2)
cv.imshow('Res 2', img_rgb)
cv.waitKey(0)