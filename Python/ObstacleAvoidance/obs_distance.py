# import and method def
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from matplotlib import cm
import time
from glob import glob
import os
import pyrealsense2.pyrealsense2 as rs    # RealSense cross-platform open-source API

# Draw closest point
def createDot(img, point):
    for i in range(-5, 6, 1):
        for j in range(-5, 6, 1):
            x = point[1] + i
            y = point[0] + j
            if x < w and y < h:
                if len(img[y,x]) == 4:
                    img[y, x] = [255,0,0, 255]  
                if len(img[y,x]) == 3:
                    img[y, x] = [255,0,0]    

def init():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device() #Stereo Module and RGB Camera
    device_product_line = str(device.get_info(rs.camera_info.product_line)) # this is D400

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        print("960x540")
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        print("640x480")
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start streaming
    colorizer = rs.colorizer()                                # Mapping depth data into RGB color space
    profile = pipeline.start(config)
    return pipeline

#img = cv2.imread('ObstacleAvoidance/images/Alfa_Laval_Sensor_301.jpg')
pipeline = init()
while True:
    start = time.time()
    # Get frame
    frameset = pipeline.wait_for_frames() 
    color_frame = frameset.get_color_frame()      
    depth_frame = frameset.get_depth_frame()                
    img = np.asanyarray(color_frame.get_data())  
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Convert image color and blur image
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    medianBlur = cv2.medianBlur(img, 5)
    # Edge detection
    median_edges = cv2.Canny(medianBlur, 50, 130)

    # Sidefill
    h, w = img.shape[:2]
    row_inds = np.indices((h, w))[0] # gives row indices in shape of img
    row_inds_at_edges = row_inds.copy()
    row_inds_at_edges[median_edges==0] = 0 # only get indices at edges, 0 elsewhere
    max_row_inds = np.amax(row_inds_at_edges, axis=0) # find the max row ind over each col

    inds_after_edges = row_inds >= max_row_inds

    filled_from_bottom = np.zeros((h, w))
    filled_from_bottom[inds_after_edges] = 255

    # Horizontal dialate and erode
    # Copy image
    horizontal = filled_from_bottom.copy()

    # Horizontal erode
    horizontal_size_erode = w // 30
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size_erode, 1))
    horizontal = cv2.erode(horizontal, horizontalStructure)

    # Horizontal dilate
    # This is done to remove small corns found in the floor
    horizontal_size_dialate = w // 20
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size_dialate, 1))
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    # 2nd horizontal erode 
    horizontal = cv2.erode(horizontal, horizontalStructure)

    # Smoothed
    converted_img = Image.fromarray(np.uint8(cm.gist_earth(horizontal)*255))
    #smoothed = converted_img.filter(ImageFilter.ModeFilter(size=13))

    # Find furthest point 
    #img_with_alpha_values = np.asarray(smoothed)
    img_with_alpha_values = np.asarray(converted_img)
    # Set robot center point
    #createDot(img_with_alpha_values, (int(h*0.9), int(w/2)))

    # Convert image to grayscale and threshold it
    bw = cv2.cvtColor(img_with_alpha_values, cv2.COLOR_RGB2GRAY)
    bw = cv2.inRange(bw, 190, 255)

    # Find all occurences of white pixels
    pixel_y, pixel_x = np.nonzero(bw)
    createDot(img_with_alpha_values, (pixel_y[0], pixel_x[0]))

    # Construct a colour image to superimpose
    mask = np.asarray(img_with_alpha_values)

    img_with_overlay = cv2.addWeighted(img, 0.9, mask[:,:,:3], 0.3, 0)
    createDot(img_with_overlay, (pixel_y[0], pixel_x[0]))
    
    
    # Add depth
    dist = depth_frame.get_distance(pixel_x[0],pixel_y[0])
    img = cv2.circle(img_with_overlay,(pixel_x[0],pixel_y[0]), radius=2, color=(0, 0, 255), thickness=3)
    cv2.putText(img_with_overlay, str("%.4f" % round(dist,4)),(pixel_x[0]+10,pixel_y[0]+30),
        cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,255),2)

    # calculate width
    dist = dist * 100
    # Convert 21850 * x^-1.033 to non-negative power
    width = 21850 * (1/(dist**1.033))
    print('width = ', width)
    left = (int(pixel_x[0] - (width/2)), pixel_y[0])
    print(left)
    right = (int(pixel_x[0] + (width/2)), pixel_y[0])
    print(right)
    # Add line
    cv2.line(img_with_overlay, left, right, (0,0,255), 2)

    # end time
    end = time.time()
    print(f"Runtime of the program is {end - start}")
    cv2.imshow('Detection', img_with_overlay)

    ch =  cv2.waitKey(0)  # Wait for for user to hit any key
    print("ch value: ", ch)
    #Save image if s (115) was pressed on keyboard
    if (ch==115):
        print("saving image")
        #cv2.imwrite(f'{path}/Distance_{start_num}.jpg',img)
        #start_num+=1
        
    
    if(ch==27):# If Escape Key was hit just exit the loop
        print("exit loop")
        break
    
    #cv2.imwrite(f'./ObstacleAvoidance/test/Alfa_Laval_Sensor_Test_With_Overlay_{number}.jpg', img_with_overlay)