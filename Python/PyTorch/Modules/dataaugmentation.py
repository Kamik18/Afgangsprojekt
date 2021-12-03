import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
import Modules.BB as bnbx

# Data Augmentation - define methods
# modified from fast.ai
def crop(im, r, c, target_r, target_c): 
    return im[r:r+target_r, c:c+target_c]

# random crop to the original size
def random_crop(x, r_pix=8):
    """ Returns a random crop"""
    r, c,*_ = x.shape
    c_pix = round(r_pix*c/r)
    rand_r = random.uniform(0, 1)
    rand_c = random.uniform(0, 1)
    start_r = np.floor(2*rand_r*r_pix).astype(int)
    start_c = np.floor(2*rand_c*c_pix).astype(int)
    return crop(x, start_r, start_c, r-2*r_pix, c-2*c_pix)

def center_crop(x, r_pix=8):
    r, c,*_ = x.shape
    c_pix = round(r_pix*c/r)
    return crop(x, r_pix, c_pix, r-2*r_pix, c-2*c_pix)

def random_cropXY(x, masks, r_pix=8):
    """ Returns a random crop"""
    new_masks = []
    r, c,*_ = x.shape
    c_pix = round(r_pix*c/r)
    rand_r = random.uniform(0, 1)
    rand_c = random.uniform(0, 1)
    start_r = np.floor(2*rand_r*r_pix).astype(int)
    start_c = np.floor(2*rand_c*c_pix).astype(int)
    xx = crop(x, start_r, start_c, r-2*r_pix, c-2*c_pix)

    for mask in masks:
        new_masks.append(crop(mask, start_r, start_c, r-2*r_pix, c-2*c_pix))
    return xx, new_masks

def transformsXY(path, bbs, transforms):
    x = cv2.imread(str(path)).astype(np.float32)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)/255
    # Select last mask
    masks = bnbx.create_masks(bbs, x)
    crazynumber = np.random.random()
    if transforms:
        for i in range(len(masks)):
            if crazynumber > 0.5:
                masks[i] = np.fliplr(masks[i]).copy()        

        if crazynumber > 0.5: 
            x = np.fliplr(x).copy()
        x, masks = random_cropXY(x, masks)
    else:
        x = center_crop(x), 
        for mask in masks:
            mask = center_crop(mask)

    return x, bnbx.mask_to_bbs(masks)

def create_corner_rect(bbs, color='red'):
    for bb in bbs:
        bb = np.array(bb, dtype=np.float32)
        plt.gca().add_patch(plt.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1], color=color,
                         fill=False, lw=3))

def show_corner_bb(im, bb):
    plt.imshow(im)
    create_corner_rect(bb)
