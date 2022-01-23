import cv2
import numpy

def BrushfireAlgorithmGrayScale(imgPath, intensityChange):
    # Setup images
    oriImage=cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE) # Read the file
    
    image = oriImage.copy()

    # Create multiple images to draw both forward and backwards iteration aswell as the final brushfire image.
    brushfireImage = image.copy()

    # Add variables for reduced namoing in the nested forloops.
    cols, rows = brushfireImage.shape[:2]
    cols = cols - 1
    rows = rows - 1

    for i in range(1,cols):
        for j in range(1, rows): 
            if brushfireImage[i,j] < 34:
                brushfireImage[i,j] = 0

    #// Brushfire image
    for i in range(1,cols):
        for j in range(1, rows): 
            left = False
            up = False
            right = False
            down = False
            present_val_fwd = brushfireImage[i,j]
            present_val_bcwd = brushfireImage[cols - i, rows - j]
            left_val = brushfireImage[i - 1,j]
            up_val = brushfireImage[i,j - 1] 
            right_val = brushfireImage[cols - i + 1, rows - j]
            down_val = brushfireImage[cols - i, rows - j + 1]

            # Check pixel to the left.
            if present_val_fwd > left_val: 
                left = True
            
            # Check the pixel above
            if present_val_fwd > up_val: 
                up = True
            
            # Determine largest value
            if left and up:
                if left_val > up_val:
                    brushfireImage[i,j] = up_val + intensityChange if up_val + intensityChange <= 255 else 255
                else:
                    brushfireImage[i,j] = left_val + intensityChange if left_val + intensityChange <= 255 else 255
            elif left:
                brushfireImage[i,j] = left_val + intensityChange if left_val + intensityChange <= 255 else 255
            elif up:
                brushfireImage[i,j] = up_val + intensityChange if up_val + intensityChange <= 255 else 255

            # Backwards iteration
            # Check the pixel to the right
            if present_val_bcwd > right_val: 
                right = True

            #+ Check the pixel below
            if present_val_bcwd > down_val:
                down = True

            if right and down:
                if right_val > down_val:
                    brushfireImage[cols-i,rows-j] = down_val + intensityChange if down_val + intensityChange <= 255 else 255
                else:
                    brushfireImage[cols-i,rows-j] = right_val + intensityChange if right_val + intensityChange <= 255 else 255
            elif right:
                brushfireImage[cols-i,rows-j] = right_val + intensityChange if right_val + intensityChange <= 255 else 255
            elif down:
                brushfireImage[cols-i,rows-j] = down_val + intensityChange if down_val + intensityChange <= 255 else 255

    return brushfireImage

def workspace(brushfire_img, original_img, threshold):
    workspace_img = original_img.copy()
    cols, rows = brushfire_img.shape[:2]
    for i in range(1,cols -1):
        for j in range(1, rows -1): 
            if brushfire_img[i,j] > threshold:
                #workspace_img[i,j] = [255,0,0]
                workspace_img[i,j] = 255
    return workspace_img

def ReduceToLowestResolution(img_input):
    img = img_input.copy()
    cols, rows = img.shape[:2]
    cols = cols - 1
    rows = rows - 1
    max = 0
    point = (0,0)
    # Forward and backwards brushfire image.
    for i in range(cols):
        for j in range(rows):
            # Check pixel to the left.
            if max < img[i,j]:
                max = img[i,j]
                point = (i,j)

    # Scale all pixels with regards to the maximum value in the image
    factor = 255 / max
    # Forward and backwards brushfire image.
    for i in range(1, cols):
        for j in range(1, rows):
            img[i,j] *= factor
    
    return img, point

def CreateMap(img):
    cols, rows = img.shape[:2]
    cols = cols - 1
    rows = rows - 1
    
    color_img = img.copy()

    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    
    #for i in range(cols):
    #    for j in range(rows):