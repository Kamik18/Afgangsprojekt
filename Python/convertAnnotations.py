from typing import NewType
import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
import glob
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

#img = cv2.imread(train_path + '//Alfa_Laval_Sensor_0.jpg')
#img = img[..., ::-1]
#b,g,r = cv2.split(img)       # get b,g,r
#img = cv2.merge([r,g,b])     # switch it to rgb
#width, height = img.shape[1], img.shape[0]
#print(width, height)
#plt.imshow(img)
#plt.show()

def CSV2YoloLine(box, width, height, classList=[]):
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

def BndBox2YoloLine(box, width, height, classList=[]):
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

    newimg = cv2.resize(img, (480,640))

    
    plt.imshow(img)
    plt.show()
    plt.imshow(newimg)
    plt.show()

# Split CSV
def split_csv():
    with open('C:\\Temp\\img temp\\csv\\test.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    print(data[0])

    line_header = (data[0][4], data[0][5], data[0][6], data[0][7])
    data.pop(0)   
    
    
    #print(name)
    for i in data:
        line_data = (i[4], i[5], i[6], i[7])
        name = i[0].replace("jpg", "csv")
        with open(f'C:\\Temp\\img temp\\test\\{name}', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(line_header)
            writer.writerow(line_data)
            f.close()



def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

import xml.etree.ElementTree as ET
def changefilename():
    import os
    firstnum = 180
    #path = 'data/OnlyOneAlfaLaval'
    img_path = 'data/temptest/images'
    input_anno_path = 'data/temp/annotations'
    output_anno_path = 'data/temptest/annotations'

    images = os.listdir(img_path)
    annos = os.listdir(input_anno_path)

    print('IMAGES:')
    for index, image in enumerate(images):
        os.rename(os.path.join(img_path, image), os.path.join(img_path, str('Alfa_Laval_Sensor_' + "{0:0=3d}".format(index+firstnum) + ".jpg")))

        
    print('ANNOTATIONS:')
    for index, anno in enumerate(annos):
        #print(index, anno)
        #print(os.path.join(anno_path, anno), os.path.join(anno_path, str('Alfa_Laval_Sensor_' + "{0:0=3d}".format(index+firstnum) + ".xml")))

        mytree = ET.parse(f'{input_anno_path}/{anno}')
        myroot = mytree.getroot()
        
        # Elements to add
        new_name = str('Alfa_Laval_Sensor_' + "{0:0=3d}".format(index+firstnum))
        new_path = f'{img_path}/{new_name}.jpg'
        new_folder = 'images'
        

        # Remove specific elements
        myroot.remove(myroot[0])
        myroot.remove(myroot[0])
        myroot.remove(myroot[0])
        
        # Add path
        filename_element = ET.Element('path')
        filename_element.text = new_path
        myroot.insert(0, filename_element)

        # Add filename
        path_element = ET.Element('filename')
        path_element.text = f'{new_name}.jpg'
        myroot.insert(0, path_element)

        # Add folder
        folder_element = ET.Element('folder')
        folder_element.text = new_folder
        myroot.insert(0, folder_element)

        indent(myroot)
        with open(f'{output_anno_path}/{new_name}.xml', 'wb') as f:
            mytree.write(f, encoding="utf-8")
        
    
        #os.rename(os.path.join(path, file), os.path.join(path, str('Alfa_Laval_Sensor_' + "{0:0=3d}".format(index+61) + ".jpg")))

        # os.rename(os.path.join(path, file), os.path.join(path, ''.join([str(index), '.jpg'])))


def extractvalues():
    file = open('data/e25_seg.txt', 'r')
    lines = file.readlines()
    
    counter = 0
    for line in lines:
        counter += 1 
        loss_index = line.find('loss')
        val_loss_index = line.find('val_loss')
        loss_find_comma = line[loss_index+5:].find(',')
        loss = line[loss_index+5:loss_index + 5 + loss_find_comma]
        val_loss = line[val_loss_index + 9:-1]
        print(counter, loss, val_loss)

def combine():
    file1 = open('data/e25_015_seg.txt', 'r')
    lines1 = file1.readlines() 

    file2 = open('data/e25_015_noseg.txt', 'r')
    lines2 = file2.readlines()

    for i in range(len(lines1)):
        line1 = lines1[i].split()
        line2 = lines2[i].split()
        print(i +1, line1[1], line1[2], line2[1], line2[2])

#extractvalues()
combine()

def overwriteFilename():
    original = [os.path.basename(x) for x in glob.glob("C:/Users/emila/Downloads/images/*.jpg")]
    #original = glob.glob("C:/Users/emila/Downloads/testing/*.jpg")
    old = glob.glob("C:/Users/emila/Downloads/images/*.jpg")

    for i in range(len(original)):
        
        new = original[i][5:]
        
        os.rename(f'C:/Users/emila/Downloads/images/{original[i]}', f'C:/Users/emila/Downloads/images/train/{new}')        
        #if i == 1:
        #    break
        #index = old[i].find("/train")
        #old_string = old[i][index+6:]
        #overwrite = original[i]
        #new_string = old[i].replace(old_string, overwrite)
        #print(old[i].replace(old_string, new_string))
        #print(new_string)
        #os.rename(old[i], new_string)
    
    
#overwriteFilename()