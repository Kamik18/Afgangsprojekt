import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
import os

# Variables
train_path = './Car_Object_Detection/images/val'
test_path = './Car_Object_Detection/images/test'

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

#print(test_path + '/vid_5_400.jpg')
#img = cv2.imread(test_path + '/vid_5_400.jpg')
#img = img[..., ::-1]
#b,g,r = cv2.split(img)       # get b,g,r
#img = cv2.merge([r,g,b])     # switch it to rgb
#width, height = img.shape[1], img.shape[0]
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

    #plt.imshow(img)

def create_labels():
    with open('./Car_Object_Detection/train_solution_bounding_boxes (1).csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                #print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                #print(f'\tType: {row[0]}, path: {row[1]}, class: {row[2]}, x_min: {row[3]}, y_min: {row[4]}, x_max: {row[5]}, y_max: {row[8]}')
                print(f'\tPath: {row[0]}, x_min: {row[1]}, y_min: {row[2]}, x_max: {row[3]}, y_max: {row[4]}')
                box = [0, float(row[1]), float(row[3]), float(row[2]), float(row[4])]
                label = BndBox2YoloLine(box, [0])
                # Convert to each single file
                with open(f'./Car_Object_Detection/labels/{row[0].replace("jpg", "txt")}', 'w') as f:
                    f.write("{} {} {} {} {}\n".format(label[0],label[1],label[2],label[3],label[4]))

                line_count += 1
        print(f'Processed {line_count} lines.')

#create_labels()

alfa_path='./Alfa_Laval/training/'
def print_img_with_annotation():
    with open('./Alfa_Laval/Labels.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            elif line_count == 1:
                print(f'Values are {", ".join(row)}')
                box = [0, float(row[2]), float(row[3]), float(row[4]), float(row[5])]
                print(alfa_path + f'{row[0]}')
                img = cv2.imread(alfa_path + f'{row[0]}')
                img = img[..., ::-1]
                b,g,r = cv2.split(img)       # get b,g,r
                img = cv2.merge([r,g,b])     # switch it to rgb
                width, height = img.shape[1], img.shape[0]
                print(width, height)
                start_pont = (int(box[0]) - int(row[1]), int(box[2]) - int(box[3]))
                print(start_pont)
                plt.imshow(img)
                plt.show()
                line_count += 1
            else:
                pass
                #print(f'\tType: {row[0]}, path: {row[1]}, class: {row[2]}, x_min: {row[3]}, y_min: {row[4]}, x_max: {row[5]}, y_max: {row[8]}')

print_img_with_annotation()