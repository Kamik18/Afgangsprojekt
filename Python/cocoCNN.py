import cv2
import numpy as np   
import time
#import pyrealsense2.pyrealsense2 as rs    # RealSense cross-platform open-source API
thres = 0.6 # Threshold to detect object

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
cap.set(10,70)
"""
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

# Skip 5 first frames to give the Auto-Exposure time to adjust
#for x in range(10):
#    pipeline.wait_for_frames() 
"""
classNames= []
classFile = 'C:\\Users\\Emil\\Documents\\GitHub\\Afgangsprojekt\\Python\\Coco\\coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split()
#print(classNames)

configPath = 'C:\\Users\\Emil\\Documents\\GitHub\\Afgangsprojekt\\Python\\Coco\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'C:\\Users\\Emil\\Documents\\GitHub\\Afgangsprojekt\\Python\\Coco\\frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

textScale = 0.7
textThickness = 2

while True:
    start = time.time()
    success,img = cap.read()
    #frameset = pipeline.wait_for_frames() 
    #img = np.asanyarray(frameset.get_color_frame().get_data())  
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    #print(classIds,bbox)


    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,color=(255,255,255),thickness=2)
            cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
            cv2.FONT_HERSHEY_COMPLEX,textScale,(255,255,255),textThickness)
            cv2.putText(img,str(round(confidence*100,2)),(box[0]+150,box[1]+30),
            cv2.FONT_HERSHEY_COMPLEX,textScale,(255,255,255),textThickness)

    cv2.imshow("Output", img)
    cv2.waitKey(1)
    end = time.time()
    print("time: ", end - start)