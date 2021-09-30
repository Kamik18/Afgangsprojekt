import atexit
import serial
from Modules import hokuyo
from Modules import serial_port
from Modules import feature
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import math
import time
import os
import numpy as np
import cv2
import smbus
import struct
import subprocess

# Gloabl variable
theta_multiplier = 1.0
robot_poses = []
laser_data = []
global_lines = []
global_points = []

# UART - Lidar communication
uart_port = '/dev/ttyACM0'
uart_speed = 19200

# I2C - Arduino communication
bus = smbus.SMBus(1)
address = 25

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
    exit(0)

# Euclidian distance from point 1 to point 2
def dist_points(point1, point2):
    Px = (point1[0] - point2[0]) ** 2
    Py = (point1[1] - point2[1]) ** 2
    return np.sqrt(Px + Py)


def in_line(point1, point2, point3):
    dist = dist_points(point1, point3)
    if (dist < 1):
        dist = 1

    if ((point2[0] - point1[0]) != 0):
        a = (point2[1] - point1[1]) / (point2[0] - point1[0])
        b = point1[1] - a * point1[0]
        return abs((a * point3[0] + b) - point3[1]) / dist
    elif ((point3[0] - point1[0]) != 0):
        a = (point3[1] - point2[1]) / (point3[0] - point2[0])
        b = point2[1] - a * point2[0]
        return abs((a * point1[0] + b) - point1[1]) / dist
    else:
        return 0


def AD2pos(distance, angle, robot_position):
    x = distance * np.cos(angle - robot_position[2]) + robot_position[0]
    y = -distance * np.sin(angle - robot_position[2]) + robot_position[1]
    return (int(x), int(y))


def convert_data_set(data, robot_position=(0, 0, 0)):
    points = []
    if not data:
        pass
    else:
        for point in data:
            length = data[point]
            if length > 20:
                coordinates = AD2pos(length, np.radians(point), robot_position)
                points.append(coordinates)
    return points

def create_map():
    global_lines = []
    global_points = []

    for index in range(0, len(laser_data)):
        pose = (robot_poses[index][0], robot_poses[index][1], robot_poses[index][2] * theta_multiplier)
        data_set = convert_data_set(laser_data[index], pose)
        data = []
        discarded_data = []
        for item in range(0, (len(data_set) - 1)):
            # Remove if closer than 150 mm
            if (dist_points(data_set[item], (0, 0)) > 150):
                if (dist_points(data_set[item], data_set[item + 1]) < 100):
                    point = ((data_set[item][0] + data_set[item + 1][0]) / 2,
                             (data_set[item][1] + data_set[item + 1][1]) / 2)
                    data.append(point)
                else:
                    data.append(data_set[item])
            else:
                discarded_data.append(data_set[item])
        global_points.append(data)

        # Find breakpoints
        lines = []
        break_point = []
        for item in range(0, len(data) - 2):
            # Validate the distance is closer than 50 mm
            if (dist_points(data[item], data[item + 1]) < 50):
                if (in_line(data[item], data[item + 1], data[item + 2]) < 0.5):
                    break_point.append(data[item])
            else:
                if (len(break_point) != 0):
                    lines.append(break_point)
                    break_point = []
        lines.append(break_point)
        global_lines.append(lines)
    
    # Plot lines
    plt.cla()
    if global_points:
        for i in range(0, len(global_points)):
            if global_points[i]:
                x, y = zip(*global_points[i])
                plt.plot(x, y, '.g')
    if global_lines:
        for i in range(0, len(global_lines)):
            if global_lines[i]:
                for j in range(0, len(global_lines[i])):
                    if global_lines[i][j]:
                        x, y = zip(*global_lines[i][j])
                        plt.plot(x, y, '-b')
    #plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.title("Lidar " + str(theta_multiplier))

    button_theta_p = Button(plt.axes([0.8, 0.025, 0.1, 0.04]), '+', color='lightgoldenrodyellow', hovercolor='0.975')
    button_theta_n = Button(plt.axes([0.2, 0.025, 0.1, 0.04]), '-', color='lightgoldenrodyellow', hovercolor='0.975')
    button_update = Button(plt.axes([0.5, 0.025, 0.1, 0.04]), 'Update', color='lightgoldenrodyellow', hovercolor='0.975')

    def minus(event):
        global theta_multiplier
        theta_multiplier -= 0.1
        print(theta_multiplier)
    button_theta_n.on_clicked(minus)

    def plus(event):
        global theta_multiplier
        theta_multiplier += 0.1
        print(theta_multiplier)
    button_theta_p.on_clicked(plus)
    
    def update(event):
        plt.close('all')
        time.sleep(1)
        create_map()
    button_update.on_clicked(update)

    plt.show()


if __name__ == '__main__':
    # Reset the encoder
    subprocess.call("./reset.sh")
    time.sleep(1)

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

    try:
        # Update last position
        last_pos = (np.inf, np.inf, np.inf)

        while (True):
            # Read state from Arduino
            data = bus.read_i2c_block_data(address, 0, 25)
            bumper_preesed = data[0]
            if (bumper_preesed):
                print("Data gathered")
                break

            x = struct.unpack('d', bytearray(data[1:9]))[0]
            y = struct.unpack('d', bytearray(data[9:17]))[0]
            theta = struct.unpack('d', bytearray(data[17:]))[0]
            
            # Update robot position
            robot_pos = (x * 1000, y * 1000, theta)

            # Check if turtle has traveled at least 25 cm
            if (dist_points(last_pos, robot_pos) > 250):
                # Append data from sensor
                laser_data.append(laser.get_single_scan())
                # Append robot position
                robot_poses.append(robot_pos)
                # Update last position
                last_pos = robot_pos

            '''
            data = bus.read_i2c_block_data(address, 0, 25)
            x = struct.unpack('d', bytearray(data[1:9]))[0]
            y = struct.unpack('d', bytearray(data[9:17]))[0]
            theta = struct.unpack('d', bytearray(data[17:]))[0] * 1.12
            
            robot_pos = (x * 1000, y * 1000, theta)
            robot_pos = (x * 1000, y * 1000, 0)

            # Check if turtle has traveled at least 25 cm
            if (dist_points(last_pos, robot_pos) > 250):
                laser_data = laser.get_single_scan()
                data_set = convert_data_set(laser_data, robot_pos)

                data = []
                discarded_data = []
                for item in range(0, (len(data_set) - 1)):
                    # Remove if closer than 150 mm
                    if (dist_points(data_set[item], (0, 0)) > 150):
                        if (dist_points(data_set[item], data_set[item + 1]) < 100):
                            point = ((data_set[item][0] + data_set[item + 1][0]) / 2,
                                     (data_set[item][1] + data_set[item + 1][1]) / 2)
                            data.append(point)
                        else:
                            data.append(data_set[item])
                    else:
                        discarded_data.append(data_set[item])
                g_points.append(data)

                # Find breakpoints
                break_point = []
                for i in range(0, len(data) - 2):
                    # Validate the distance is closer than 50 mm
                    if (dist_points(data[i], data[i + 1]) < 50):
                        if (in_line(data[i], data[i + 1], data[i + 2]) < 0.5):
                            break_point.append(data[i])
                    else:
                        if (len(break_point) != 0):
                            lines.append(break_point)
                            break_point = []
                lines.append(break_point)
                poses.append(robot_pos)

                # Increment number of detections
                count += 1

                # Update last position
                last_pos = robot_pos

            if (count > 10):
                break
            
            # Wait a second
            time.sleep(1)
            #'''
            '''
            laser_data = laser.get_single_scan()
            data_set = convert_data_set(laser_data, (0, 0, 0))
            t0 = time.process_time()

            data = []
            discarded_data = []
            for item in range(0, (len(data_set) - 1)):
                # Remove if closer than 150 mm
                if (dist_points(data_set[item], (0, 0)) > 150):
                    if (dist_points(data_set[item], data_set[item + 1]) < 100):
                        point = ((data_set[item][0] + data_set[item + 1][0]) / 2,
                                 (data_set[item][1] + data_set[item + 1][1]) / 2)
                        data.append(point)
                    else:
                        data.append(data_set[item])
                else:
                    discarded_data.append(data_set[item])
            g_points = data

            # Find breakpoints
            lines = []
            break_point = []
            for i in range(0, len(data) - 2):
                # Validate the distance is closer than 50 mm
                if (dist_points(data[i], data[i + 1]) < 50):
                    if (in_line(data[i], data[i + 1], data[i + 2]) < 0.5):
                        break_point.append(data[i])
                else:
                    if (len(break_point) != 0):
                        lines.append(break_point)
                        break_point = []
            lines.append(break_point)

            print("Elapsed time: ", time.process_time() - t0)

            plt.cla()
            if data:
                x, y = zip(*data)
                plt.plot(x, y, '.g', label="Acceptable")
            if lines:
                for i in range(0, len(lines)):
                    if lines[i]:
                        x, y = zip(*lines[i])
                        plt.plot(x, y, '-b')
            plt.legend()
            plt.axis("equal")
            plt.grid(True)
            plt.title("Lidar")
            plt.pause(0.5)
            #'''

    except KeyboardInterrupt:
        print("Interrupted")
        # Terminate session
        goodbye() 
    
    '''
    for index in range(0, len(laser_data)):
        data_set = convert_data_set(laser_data[index], robot_poses[index])
        data = []
        discarded_data = []
        for item in range(0, (len(data_set) - 1)):
            # Remove if closer than 150 mm
            if (dist_points(data_set[item], (0, 0)) > 150):
                if (dist_points(data_set[item], data_set[item + 1]) < 100):
                    point = ((data_set[item][0] + data_set[item + 1][0]) / 2,
                             (data_set[item][1] + data_set[item + 1][1]) / 2)
                    data.append(point)
                else:
                    data.append(data_set[item])
            else:
                discarded_data.append(data_set[item])
        global_points.append(data)
        # Find breakpoints
        lines = []
        break_point = []
        for item in range(0, len(data) - 2):
            # Validate the distance is closer than 50 mm
            if (dist_points(data[item], data[item + 1]) < 50):
                if (in_line(data[item], data[item + 1], data[item + 2]) < 0.5):
                    break_point.append(data[item])
            else:
                if (len(break_point) != 0):
                    lines.append(break_point)
                    break_point = []
        lines.append(break_point)
        global_lines.append(lines)

        # Print lines and data
        plt.cla()
        if data:
            x, y = zip(*data)
            plt.plot(x, y, '.g', label="Acceptable")
        if lines:
            for item in range(0, len(lines)):
                if lines[item]:
                    x, y = zip(*lines[item])
                    plt.plot(x, y, '-b')
        plt.legend()
        plt.axis("equal")
        plt.grid(True)
        plt.title("Lidar: " + str(index + 1) + " of " + str(len(laser_data)))
        #plt.show()
        plt.pause(0.5)
        #'''


    create_map()

    # Terminate session
    goodbye()    
