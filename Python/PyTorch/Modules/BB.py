import cv2
import numpy as np
import pandas as pd

# Define methods for resizing and bounding boxes
def read_image(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)

def create_masks(bb, x):
    """Creates a mask for the bounding box of same shape as image"""
    rows,cols,*_ = x.shape
    ## Create an image for each bounding box
    Y = np.zeros((rows, cols))
    i = 0
    images = []
    for bndbox in bb:
        images.append(np.zeros((rows,cols)))
        bndbox = bndbox.astype(np.int)
        images[i][bndbox[1]:bndbox[3], bndbox[0]:bndbox[2]] = 1.
        Y[bndbox[1]:bndbox[3], bndbox[0]:bndbox[2]] = 1.
        i += 1
    images.append(Y)
    return images

def mask_to_bbs(images):
    """Convert mask Y to a bounding box, assumes 0 as background nonzero object"""
    bbs = []
    for Y in images[:-1]:
        rows, cols = np.nonzero(Y)
        if len(cols)==0: 
            #bbs.append(np.zeros(4, dtype=np.float32))
            bbs.append(np.zeros(4))
            continue
        top_row = np.min(rows)
        left_col = np.min(cols)
        bottom_row = np.max(rows)
        right_col = np.max(cols)
        bbs.append(np.array([left_col, top_row, right_col, bottom_row]))
        #bbs.append(np.array([left_col, top_row, right_col, bottom_row], dtype=np.float32))
    return bbs

def create_bbs_array(x):
    """Generates array of bounding boxes from a train_df row"""
    array = []
    for item in x[3]:
        if not pd.isna(item):
            array.append(np.array([item['xmin'],item['ymin'],item['xmax'],item['ymax']]))
    return array

def resize_image_bb(read_path,write_path,bbs,sz):
    """Resize an image and its bounding box and write image to new path"""
    im = read_image(read_path)
    #im_resized = cv2.resize(im, (int(1.49*sz), sz))
    masks = []
    for mask in create_masks(bbs, im):
        masks.append(cv2.resize(mask, (int(1.49*sz), sz)))
    
    new_path = str(write_path/read_path.parts[-1])
    #cv2.imwrite(new_path, cv2.cvtColor(im_resized, cv2.COLOR_RGB2BGR))
    return new_path, mask_to_bbs(masks)