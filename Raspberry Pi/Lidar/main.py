import atexit
import serial
from Modules import hokuyo
from Modules import serial_port
from Modules import feature
import matplotlib.pyplot as plt
import math
import time
import os
import numpy as np
import cv2


def goodbye():
    print('get_version_info')
    print(laser.get_version_info())
    print('get_sensor_specs')
    print(laser.get_sensor_specs())
    print('reset')
    print(laser.reset())
    print('laser_off')
    print(laser.laser_off())
    print('Complete')


uart_port = '/dev/ttyACM0'
uart_speed = 19200

featureDetect = feature.featureDetection()

if __name__ == '__main__':
    laser_serial = serial.Serial(
        port=uart_port, baudrate=uart_speed, timeout=0.5)
    port = serial_port.SerialPort(laser_serial)

    laser = hokuyo.Hokuyo(port)

    print('laser_on')
    print(laser.laser_on())
    print(laser.set_high_sensitive(True))
    print(laser.set_motor_speed())

    print("Fetch data")
    t0 = time.process_time()
    for i in range(100):
        data = laser.get_single_scan()
    time_elapsed = time.process_time() - t0
    print("Elapsed time: ", time_elapsed,
          " frequency: ", 1/(time_elapsed / (i+1)))

    data = laser.get_single_scan()
    featureDetect.laser_points_set(data)

    BREAK_POINT_IND = 0
    try:
        while (True):
            data = laser.get_single_scan()
            featureDetect.laser_points_set(data)
            data_file = open("data.dat", "w")
            for element in featureDetect.LASERPOINTS:
                data_file.write(str(element))
            data_file.close()

            points_x, points_y = zip(*featureDetect.LASERPOINTS)
            print(len(points_x))
            min_x, max_x = abs(min(points_x)), abs(max(points_x))
            min_y, max_y = abs(min(points_y)), abs(max(points_y))
            print("min_x: ", min_x)
            print("min_y: ", min_x)
            print("max_x: ", max_x)
            print("max_y: ", max_y)

            edgesMat = np.zeros(
                (max_y + min_y + 1, max_x + min_x + 1), dtype="uint8")

            for point in featureDetect.LASERPOINTS:
                x = point[0] + min_x
                y = point[1] + min_y
                edgesMat[y, x] = 255

            # Apply edge detection method on the image
            edgesCanny = cv2.Canny(edgesMat, 50, 150, apertureSize=3)
            plt.figure()
            plt.imshow(edgesMat)

            # This returns an array of r and theta values
            lines = cv2.HoughLines(edgesCanny, 1, np.pi/180, 200)
            print(lines)

            img = cv2.imread('Lines.png')

            # Convert the img to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Apply edge detection method on the image
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            plt.figure()
            plt.imshow(edges)
            plt.show()

            # This returns an array of r and theta values
            lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

            # The below for loop runs till r and theta values
            # are in the range of the 2d array
            for r, theta in lines[0]:

                # Stores the value of cos(theta) in a
                a = np.cos(theta)

                # Stores the value of sin(theta) in b
                b = np.sin(theta)

                # x0 stores the value rcos(theta)
                x0 = a*r

                # y0 stores the value rsin(theta)
                y0 = b*r

                # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
                x1 = int(x0 + 1000*(-b))

                # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
                y1 = int(y0 + 1000*(a))

                # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
                x2 = int(x0 - 1000*(-b))

                # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
                y2 = int(y0 - 1000*(a))

                # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
                # (0,0,255) denotes the colour of the line to be
                # drawn. In this case, it is red.
                cv2.line(img, (x1, y1), (x2, y2), (128), 2)

            h, w = edgesMat.shape

            plt.figure()
            plt.imshow(img)
            plt.figure()
            plt.imshow(edgesMat)
            plt.show()

            '''
            t0 = time.process_time()
            BREAK_POINT_IND = 0
            endpoints = [0, 0]
            PREDICTED_POINTS_TODRAW = []
            lines = []

            while (BREAK_POINT_IND < (featureDetect.NP - featureDetect .PMIN)):
                seedSeg = featureDetect.seed_segment_detection(BREAK_POINT_IND)
                if (seedSeg == False):
                    print("Seed error")
                    break
                else:
                    seedSegment = seedSeg[0]
                    PREDICTED_POINTS_TODRAW = seedSeg[1]
                    INDICES = seedSeg[2]
                    results = featureDetect.seed_segment_growing(
                        INDICES, BREAK_POINT_IND)
                    if (results == False):
                        BREAK_POINT_IND = INDICES[1]
                        continue
                    else:
                        line_eq = results[1]
                        m, c = results[5]
                        line_seq = results[0]
                        OUTERMOST = results[2]
                        BREAK_POINT_IND = results[3]

                        endpoints[0] = featureDetect.projection_point2line(
                            OUTERMOST[0], m, c)
                        endpoints[1] = featureDetect.projection_point2line(
                            OUTERMOST[1], m, c)

                        lines.append(endpoints.copy())

            print("Number of lines: ", len(lines))
            print("ENDPOINTS: ", lines)
            print("Elapsed time: ", time.process_time() - t0)
            # '''

            plt.cla()
            points_x, points_y = zip(*featureDetect.LASERPOINTS)
            plt.plot(points_x, points_y, ".", label="points")
            # for i in range(len(lines)):
            #    x = []
            #    y = []
            #    for j in range(len(lines[0])):
            #        x.append(lines[i][j][0])
            #        y.append(lines[i][j][1])
            #    plt.plot(x, y)
            plt.legend()
            plt.axis("equal")
            plt.grid(True)
            plt.title("Lidar")
            plt.show()
            break

    except KeyboardInterrupt:
        print("Interrupted")
        print('get_version_info')
        print(laser.get_version_info())
        print('get_sensor_specs')
        print(laser.get_sensor_specs())
        print('reset')
        print(laser.reset())
        print('laser_off')
        print(laser.laser_off())
        print('Complete')
        exit(0)
