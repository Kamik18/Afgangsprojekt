from Modules.utils import compute_traj
from Modules.lqr_control import *
from Modules import hokuyo
from Modules import serial_port
from Modules import feature
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import subprocess
import time
import atexit
import serial
import math
import time
import os
import numpy as np
import cv2
import smbus
import struct
import subprocess

# Gloabl variable
robot_poses = []
laser_data = []
theta_multiplier = 1.12

global_lines = []
global_points = []

# Create the track waypoints
goal = [(0.0, 0.0), (0.0, 0.25), (0.0, 0.5)
        (0.0, 0.5), (0.25, 0.5), (0.5, 0.5), (0.75, 0.5),
        (1.0, 0.5), (1.25, 0.5), (1.5, 0.5), (1.75, 0.5),
        (2.0, 0.5), (2.25, 0.5), (2.5, 0.5), (2.75, 0.5),
        (3.0, 0.5), (3.25, 0.5), (3.5, 0.5), (3.75, 0.5),
        (4.0, 0.5), (4.25, 0.5), (4.5, 0.5), (4.75, 0.5),
        (5.0, 0.5)]
ax = []
ay = []
trajectory = []

# UART - Lidar communication
uart_port = '/dev/ttyACM0'
uart_speed = 19200

# I2C - Arduino communication
bus = smbus.SMBus(1)
address = 25

# Euclidian distance from point 1 to point 2


def dist_points(point1, point2):
    Px = (point1[0] - point2[0]) ** 2
    Py = (point1[1] - point2[1]) ** 2
    return np.sqrt(Px + Py)


def in_line(point1, point2, point3):
    dist = dist_points(point1, point3)
    if (0 == dist):
        dist = 0.01

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
    discarded_data = []

    for index in range(0, len(laser_data)):
        pose = (robot_poses[index][0],
                robot_poses[index][1],
                robot_poses[index][2] * theta_multiplier)
        data_set = convert_data_set(laser_data[index], pose)
        data = []
        for item in range(0, (len(data_set) - 1)):
            # Remove if not within 150 mm and 2.5 m
            if (150 < dist_points(data_set[item], (pose[0], pose[1])) < 2500):
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
            if (dist_points(data[item], data[item + 1]) < 25):
                if (in_line(data[item], data[item + 1], data[item + 2]) < 0.25):
                    break_point.append(data[item])
            else:
                if (len(break_point) != 0):
                    lines.append(break_point)
                    break_point = []
        lines.append(break_point)
        global_lines.append(lines)

    # Plot lines
    plt.cla()
    if global_lines:
        for i in range(0, len(global_lines)):
            if global_lines[i]:
                for j in range(0, len(global_lines[i])):
                    if global_lines[i][j]:
                        x, y = zip(*global_lines[i][j])
                        plt.plot(x, y, '-b')
    # plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.title("Lidar")
    plt.show()


def reset_encoder():
    subprocess.call("./../reset.sh")
    time.sleep(2)


def goodbye():
    # Stop the PIDs
    bus.write_i2c_block_data(address, 0, [0, 0, 0])

    # Append goal point
    ax.append(goal[0])
    ay.append(goal[1])

    # Compute the desired trajectory
    desired_traj = compute_traj(ax, ay)

    # Display the trajectory that the mobile robot executed
    plt.close()
    flg, _ = plt.subplots(1)
    plt.plot(desired_traj[:, 0], desired_traj[:, 1], "-b", label="Desired")
    for i in range(len(trajectory)):
        plt.plot(trajectory[i][:, 0], trajectory[i][:, 1], "-g")
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.legend()
    plt.show()

    # Print traveled distance
    data = bus.read_i2c_block_data(address, 0, 25)
    x = struct.unpack('d', bytearray(data[1:9]))[0]
    y = struct.unpack('d', bytearray(data[9:17]))[0]
    angle = struct.unpack('d', bytearray(data[17:]))[0]
    print("x: ", x)
    print("y: ", y)
    print("angle: ", angle)

    # Create a map
    create_map()

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


atexit.register(goodbye)

if __name__ == '__main__':
    print("LQR steering control tracking start")

    # Starting the laser
    laser_serial = serial.Serial(
        port=uart_port, baudrate=uart_speed, timeout=0.5)
    port = serial_port.SerialPort(laser_serial)

    laser = hokuyo.Hokuyo(port)

    print('laser_on')
    print(laser.laser_on())
    print(laser.set_high_sensitive(True))
    print(laser.set_motor_speed())

    # Reset the encoders
    reset_encoder()

    # Append data from sensor
    laser_data.append(laser.get_single_scan())
    # Append robot position
    robot_poses.append([0.0, 0.0, 0.0])

    for target in goal:
        pos, delta_trajectory = closed_loop_prediction(target)
        trajectory.append(delta_trajectory)
        ax.append(target[0])
        ay.append(target[1])

        # Update robot position
        robot_pos = (pos[0] * 1000, pos[1] * 1000, pos[2])
        # Append data from sensor
        laser_data.append(laser.get_single_scan())
        # Append robot position
        robot_poses.append(robot_pos)

    # Stop the PIDs
    bus.write_i2c_block_data(address, 0, [0, 0, 0])
