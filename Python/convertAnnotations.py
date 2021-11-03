import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
import os

# Variables
train_path = './Alfa_Laval/images/train/'
test_path = './Alfa_Laval/images/test/'

def split_train_into_train_val():
    labels_path = './Car_Object_Detection/labels/val'

    files_labels = os.listdir(labels_path)
    labels = []
    for f in files_labels:
        labels.append(f.replace(".txt", ""))

    files_images = os.listdir(train_path)
    images = []
    for f in files_images:
        images.append(f.replace(".jpg", ""))

    for i in labels:
        #print(i)
        if i in images:
            pass
            #print("ok")
        else:
            print("Not found")

img = cv2.imread(train_path + '//Alfa_Laval_Sensor_0.jpg')
#img = img[..., ::-1]
b,g,r = cv2.split(img)       # get b,g,r
img = cv2.merge([r,g,b])     # switch it to rgb
width, height = img.shape[1], img.shape[0]
#print(width, height)
#plt.imshow(img)
#plt.show()

def CSV2YoloLine(box, classList=[]):
    print(box)
    xtopleft = box[1]
    ytopleft = box[2]
    w = float(box[3]) / width 
    h = float(box[4]) / height
    xcen = float(xtopleft + w / 2) / width
    ycen = float(ytopleft + h / 2) / height

    # PR387
    boxName = box[0]
    if boxName not in classList:
        classList.append(boxName)

    classIndex = classList.index(boxName)
    print(classIndex, xcen, ycen, w, h)
    return classIndex, xcen, ycen, w, h

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

    #plt.imshow(img)

def create_labels():
    with open('./Alfa_Laval/TestLabels.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                #print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                #print(f'\tType: {row[0]}, path: {row[1]}, class: {row[2]}, x_min: {row[3]}, y_min: {row[4]}, x_max: {row[5]}, y_max: {row[8]}')
                print(f'\tPath: {row[0]}, label: {row[1]}, x_topleft: {row[2]}, y_topleft: {row[3]}, x_width: {row[4]}, y_height: {row[5]}')
                box = [0, float(row[2]), float(row[3]), float(row[4]), float(row[5])]
                label = CSV2YoloLine(box, [0])
                
                #print(train_path + f'{row[0]}')
                #img = cv2.imread(train_path + f'{row[0]}')
                #img = img[..., ::-1]
                #b,g,r = cv2.split(img)       # get b,g,r
                #img = cv2.merge([r,g,b])     # switch it to rgb
                #width, height = img.shape[1], img.shape[0]
                #print(width, height)
                #x_start = int(label[1] - (label[3] / 2 ))
                #y_start = int(label[2] - (label[4] / 2 ))
                #x_end = int(label[1] + (label[3] / 2 ))
                #y_end = int(label[2] + (label[4] / 2 ))
                #start_point=(x_start, y_start)
                #end_point=(x_end, y_end)
                #print(start_point)
                #print(end_point)
                #img = cv2.rectangle(img, start_point, end_point, (255,0,0), 2)
                #plt.imshow(img)
                #plt.show()

                # Convert to each single file
                with open(f'./Alfa_Laval/labels/{row[0].replace("jpg", "txt")}', 'w') as f:
                    f.write("{} {} {} {} {}\n".format(label[0],label[1],label[2],label[3],label[4]))

                line_count += 1
        print(f'Processed {line_count} lines.')

#create_labels()

alfa_path='./Alfa_Laval/images/test/'
def print_img_with_annotation():
    with open('./Alfa_Laval/TestLabels.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            #else:
            elif line_count == 12:
                print(f'Values are {", ".join(row)}')
                box = [0, float(row[2]), float(row[3]), float(row[4]), float(row[5])]
                print(alfa_path + f'{row[0]}')
                img = cv2.imread(alfa_path + f'{row[0]}')
                #img = img[..., ::-1]
                #b,g,r = cv2.split(img)       # get b,g,r
                #img = cv2.merge([r,g,b])     # switch it to rgb
                width, height = img.shape[1], img.shape[0]
                print(width, height)
                x_start = int(box[1])
                y_start = int(box[2])
                x_end = int(box[1] + box[3])
                y_end = int(box[2] + box[4])
                start_point=(x_start, y_start)
                end_point=(x_end, y_end)
                print(start_point)
                print(end_point)
                img = cv2.rectangle(img, start_point, end_point, (255,0,0), 2)
                #start_point = ((int(box[0]) - int(row[1])), (int(box[2]) - int(box[3])))
                plt.imshow(img)
                plt.show()
                line_count += 1
            else:
                line_count += 1
                #print(f'\tType: {row[0]}, path: {row[1]}, class: {row[2]}, x_min: {row[3]}, y_min: {row[4]}, x_max: {row[5]}, y_max: {row[8]}')

#pr1int_img_with_annotation()
path='./Alfa_Laval/images/test/'
def resize_img():
    testdir = os.listdir(path)
    print(f'{path}{testdir[0]}')
    img = cv2.imread(path + f'{testdir[0]}')

    newimg = cv2.resize(img, (480,480))

    
    plt.imshow(img)
    plt.show()
    plt.imshow(newimg)
    plt.show()

resize_img()