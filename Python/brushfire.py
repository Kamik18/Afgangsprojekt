import cv2

oriImage = cv2.imread("./ICP/Kort_binaer.png")


brushfireImage = oriImage.copy()
brushfireImageForward = oriImage.copy()
brushfireImageReverse = oriImage.copy()

rows, cols, _ = brushfireImage.shape

print(rows)
print(cols)