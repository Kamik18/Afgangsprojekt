"""
Installing Tensorflow with CUDA, cuDNN and GPU support on Windows 10
https://towardsdatascience.com/installing-tensorflow-with-cuda-cudnn-and-gpu-support-on-windows-10-60693e46e781
Verion compatability: https://www.tensorflow.org/install/source#gpu
Install CUDA 11.5 - https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html
Install cuDNN 8.1 - https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#troubleshoot-windows



import subprocess
import sys
def install_package(package):
    print("Installing: " + package)
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_package("tensorflow==2.5.0")
install_package("tensorflow-gpu==2.5.0")
install_package("tf-nightly")
install_package("numpy")
install_package("tflite-model-maker")

# use pip check to verify that versions work together.
"""

import time
import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5/bin")
#import tensorflow as tf
#from tflite_model_maker.config import ExportFormat
#from tflite_model_maker import model_spec
#from tflite_model_maker import object_detector
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

#assert tf.__version__.startswith('2')

#tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

# Variables
train_path = './Car_Object_Detection/data/training_images'
test_path = './Car_Object_Detection/data/testing_images'


# Program start
from os import walk
# Load the raw car detection model data
def somemethod(path):
    images = []
    for (dirpath, dirnames, filenames) in walk(path):
        for str in filenames:
            images.append(str)
    return images

#start = time.time()
#data = somemethod(train_path)
#print(len(data))
# Follow 70/30 rule. 70% for training and 30% for validation
#split = int((len(data)-(((len(data)/100)*30))))
#print(split)
#train_data = data[:split]
#validation_data = data[split:]
#print("train legth: ", len(train_data))
#print("val legth: ", len(validation_data))
#test_data = somemethod(test_path)
#end = time.time()
#print(end - start)
#print("Train_mages length: ", len(train_data))
#print("Train_mages length: ", len(validation_data))
#print("Test_images length: ", len(test_data))
#img = cv2.imread(train_path + '/vid_4_600.jpg')
#img = img[..., ::-1]
#b,g,r = cv2.split(img)       # get b,g,r
#img = cv2.merge([r,g,b])     # switch it to rgb
#width, height = img.shape[1], img.shape[0]
#print(width, height)
#plt.imshow(img)

#spec = model_spec.get('efficientdet_lite0')
#import pandas as pd
#train_sub = pd.read_csv('./Car_Object_Detection/data/train_solution_bounding_boxes (1).csv')
#google_sub = pd.read_csv('./Car_Object_Detection/data/train_solution_bounding_boxes.csv')
#print(train_sub.head())
#print(google_sub.head(3))

#print(google_sub.describe())
#train_data, validation_data, test_data = object_detector.DataLoader.from_csv('./Car_Object_Detection/data/test.csv')
#print("Train: ", len(train_data))
#print(train_data)
#print("Val: ", len(validation_data))
#print("Test: ", len(test_data))

#print("Loaded data")

#model = object_detector.create(train_data, model_spec=spec, batch_size=8, train_whole_model=True, validation_data=validation_data)
#print("load done")

#model.evaluate(test_data)
#model.export(export_dir='.')


img = cv2.imread(test_path + '/vid_5_400.jpg')
img = img[..., ::-1]
b,g,r = cv2.split(img)       # get b,g,r
img = cv2.merge([r,g,b])     # switch it to rgb
width, height = img.shape[1], img.shape[0]
#print(width, height)

def BndBox2YoloLine(box, classList=[]):
    xmin = box[1]
    xmax = box[2]
    ymin = box[3]
    ymax = box[4]
    xcen = float((xmin + xmax)) / 2 / width
    ycen = float((ymin + ymax)) / 2 / height

    w = float((xmax - xmin)) / width
    h = float((ymax - ymin)) / height

    # PR387
    boxName = box[0]
    if boxName not in classList:
        classList.append(boxName)

    classIndex = classList.index(boxName)
    return classIndex, xcen, ycen, w, h

import csv


#plt.imshow(img)


with open('./Car_Object_Detection/data/test.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            #print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            #print(f'\tType: {row[0]}, path: {row[1]}, class: {row[2]}, x_min: {row[3]}, y_min: {row[4]}, x_max: {row[7]}, y_max: {row[8]}')
            box = [0, float(row[3]), float(row[7]), float(row[4]), float(row[7])]
            ss = BndBox2YoloLine(box, [0])
            # Convert to each single file
            with open('./Car_Object_Detection/data/readme.txt', 'a') as f:
                f.write("{},{},{},{},{}\n".format(ss[0],ss[1],ss[2],ss[3],ss[4]))

            line_count += 1
    print(f'Processed {line_count} lines.')

