import cv2
import numpy as np   
import time
import sys
import pyrealsense2.pyrealsense2 as rs    # RealSense cross-platform open-source API

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
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
colorizer = rs.colorizer()                                # Mapping depth data into RGB color space
profile = pipeline.start(config)

textScale = 0.7
textThickness = 2
start_point = (100,250)
end_point = (520,450)


try:
  while True:
      start = time.time()
      frameset = pipeline.wait_for_frames() 
      color_frame = frameset.get_color_frame()      
      depth_frame = frameset.get_depth_frame()                
      img = np.asanyarray(color_frame.get_data())  
      depth_image = np.asanyarray(depth_frame.get_data())
      
      # Crop depth data:
      depth = img[start_point[1]:end_point[1],start_point[0]:end_point[0]]
      print(depth.shape)
      
      closest_value = sys.maxsize
      closest_point = [0,0]
      furthest_value = 0
      furthest_point = [0,0]
      #for i in range(depth.shape[1]):
      #    for j in range(depth.shape[0]):
      #        if 0 < depth_image[j][i] > closest_value:
      #            closest_value = depth_image[j][i]
      #            closest_point = [i,j]
      
      #closest_point[0] = start_point[0] + closest_point[0] 
      #closest_point[1] = start_point[1] + closest_point[1]
      for i in range(start_point[1], end_point[1]):
          for j in range(start_point[0], end_point[0]):
            if 0 < depth_image[i][j] < closest_value:
                closest_value = depth_image[i][j]
                closest_point = [j,i]
            if 0 < depth_image[i][j] > furthest_value:
                furthest_value = depth_image[i][j]
                furthest_point = [j,i]
      print("Closest point = ", closest_point, ", value = ", closest_value, ", Distance = ", depth_frame.get_distance(closest_point[0], closest_point[1]), " m")
      print("Furthest point = ", furthest_point, ", value = ", furthest_value, ", Distance = ", depth_frame.get_distance(furthest_point[0], furthest_point[1]), " m")
      
      
      width = color_frame.get_width()
      height = color_frame.get_height()
      depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
      #depth_middle = depth_frame.get_distance(int(width/2), int(height/2)) * depth_scale
      depth_middle = depth_image * depth_scale
      dist,_,_,_ = cv2.mean(depth_middle) 
      print("Depth_middle", dist)


      print("middle (scaled) = ", int(width/2), int(height/2), ", Distance = ", dist, " m")
      print("middle point = ", int(width/2), int(height/2), ", Distance = ", depth_frame.get_distance(int(width/2), int(height/2)), " m")
      
      
      # Printing Circles
      img = cv2.circle(img,(int(width/2),int(height/2)), radius=2, color=(0, 0, 255), thickness=3)
      cv2.putText(img,"Midt",(int(width/2)+10,int(height/2)+30),
      cv2.FONT_HERSHEY_COMPLEX,textScale,(0,0,255),textThickness)
      img = cv2.circle(img,closest_point, radius=2, color=(0, 0, 255), thickness=3)
      cv2.putText(img,"Closest",(closest_point[0]+10,closest_point[1]+30),
      cv2.FONT_HERSHEY_COMPLEX,textScale,(0,0,255),textThickness)
      img = cv2.circle(img,furthest_point, radius=2, color=(0, 255, 255), thickness=3)
      cv2.putText(img,"furthest",(furthest_point[0]+10,furthest_point[1]+30),
      cv2.FONT_HERSHEY_COMPLEX,textScale,(0,255,255),textThickness)
      #img = cv2.circle(img,(640,480), radius=3, color=(0, 0, 255), thickness=4)
      #img = cv2.circle(img,(0,0), radius=3, color=(0, 0, 255), thickness=4)
      cv2.rectangle(img, start_point, end_point,color=(255,255,255),thickness=2)
      depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)       
      cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
      cv2.resizeWindow('Output', 640,480)
      cv2.moveWindow('Output', 900,0)
      cv2.imshow("Output", img)
      cv2.namedWindow('depth', cv2.WINDOW_NORMAL)
      cv2.resizeWindow('depth', 640,480)
      cv2.moveWindow('depth', 900,500)
      cv2.imshow("depth", depth_colormap)
      #cv2.waitKey(1)
      #cv2.namedWindow('Crop', cv2.WINDOW_NORMAL)
      #cv2.resizeWindow('Crop', 420,200)
      #cv2.moveWindow('Crop', 800,1400)
      #cv2.imshow("Crop", depth)

      cv2.waitKey(1)
      end = time.time()
    
      time_convert(end - start)

except KeyboardInterrupt:
    cv2.destroyAllWindows()