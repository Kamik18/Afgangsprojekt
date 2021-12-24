import cv2
import numpy as np   
import time
import sys
import os
import pyrealsense2.pyrealsense2 as rs    # RealSense cross-platform open-source API
#import pyrealsense2 as rs    # RealSense cross-platform open-source API

def time_convert(sec):
  mins = sec // 60
  sec = round(sec % 60, 2)
  hours = mins // 60
  mins = mins % 60
  print("Time Lapsed = {0}:{1}:{2}".format(int(hours),int(mins),sec))

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

# Start streaming
colorizer = rs.colorizer()                                # Mapping depth data into RGB color space
profile = pipeline.start(config)

textScale = 0.7 
textThickness = 2
start_point = (100,250)
end_point = (520,450)

path='ObstacleAvoidance/'
start_num=0

try:
    while True:
        start = time.time()
        frameset = pipeline.wait_for_frames() 
        color_frame = frameset.get_color_frame()      
        depth_frame = frameset.get_depth_frame()                
        img = np.asanyarray(color_frame.get_data())  
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # Crop depth data:
        depth = img[start_point[1]:end_point[1],start_point[0]:end_point[0]]
        print(depth.shape)
    
        width = color_frame.get_width()
        height = color_frame.get_height()
        
        print("middle point = ", int(width/2), int(height/2), ", Distance = ", depth_frame.get_distance(int(width/2), int(height/2)), " m")
        
        # Printing Circles
        x = int(width/2)
        y = int(height/2)
        img = cv2.circle(img,(x,y), radius=2, color=(0, 0, 255), thickness=3)
        dist = depth_frame.get_distance(x,y)
        cv2.putText(img, str("%.4f" % round(dist,4)),(x+10,y+30),
        cv2.FONT_HERSHEY_COMPLEX,textScale,(0,0,255),textThickness)
        #cv2.rectangle(img, start_point, end_point,color=(255,255,255),thickness=2)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)   
        
        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape
        
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Output', 640,480)
        cv2.moveWindow('Output', 260,0)
        cv2.imshow("Output", img)
        cv2.namedWindow('depth', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('depth', 640,480)
        cv2.moveWindow('depth', 900,0)
        cv2.imshow("depth", depth_colormap)





        ch =  cv2.waitKey(0)  # Wait for for user to hit any key
        print("ch value: ", ch)
        #Save image if s (115) was pressed on keyboard
        if (ch==115):
           print("saving image")
           #new = fr'{path}\Alfa_Laval_Sensor_{start_num}.jpg'
           #print(new)
           cv2.imwrite(f'{path}/Distance_{start_num}.jpg',img)
           start_num+=1
           
        
        if(ch==27):# If Escape Key was hit just exit the loop
            print("exit loop")
            break

        end = time.time()
        
        time_convert(end - start)

except KeyboardInterrupt:
    cv2.destroyAllWindows()