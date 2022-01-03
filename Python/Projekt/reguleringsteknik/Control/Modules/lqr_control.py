# Import important libraries
import numpy as np
import matplotlib.pyplot as plt
from Modules.kinematics import *
import math
import smbus
import struct
    
# I2C - Arduino communication
bus = smbus.SMBus(1)
address = 25

serial_debug = False

# Euclidian distance from point 1 to point 2
def dist_points(point1, point2):
    Px = (point1[0] - point2[0]) ** 2
    Py = (point1[1] - point2[1]) ** 2
    return np.sqrt(Px + Py)
    
def constrain(val, min_val, max_val):
    if val < min_val: return min_val
    if val > max_val: return max_val
    return val

def closed_loop_prediction(target_pos, next_target_pos, pos = (0.0, 0.0, 0.0), offset = (0.0, 0.0, 0.0)):
    # Simulation Parameters
    goal_dist = 0.15  # How close we need to get to the goal    
    dt = 0.05  # Timestep interval

    # Initial state of the car
    state = (pos[0] + offset[0], pos[1] + offset[1], pos[2] + offset[2])

    # Get the Cost-to-go and input cost matrices for LQR
    Q = get_Q()  # Defined in kinematics.py
    R = get_R()  # Defined in kinematics.py

    # Initialize the Car and the Car's landmark sensor
    DiffDrive = DifferentialDrive()

    # Create objects for storing states and estimated state
    traj = np.array([state])

    prev_dist = np.inf
    while True:
        # Generate optimal control commands
        u_lqr = dLQR(DiffDrive, Q, R, state, target_pos[0:3], dt)
        
        # Set motor speed
        cmd = [0, 0, 0]
        cmd[0] = constrain(int(abs(u_lqr[0]) * 1000), 0, 255)
        if (u_lqr[1] > 0):
            cmd[1] = constrain(int(abs(u_lqr[1]) * 1000), 0, 255)
        else:
            cmd[2] = constrain(int(abs(u_lqr[1]) * 1000), 0, 255)
        bus.write_i2c_block_data(address, 0, cmd)

        # Add sensors and update position
        # Move forwad in time
        state = [0.0, 0.0, 0.0]
        data = bus.read_i2c_block_data(address, 0, 25)
        x = struct.unpack('d', bytearray(data[1:9]))[0]
        y = struct.unpack('d', bytearray(data[9:17]))[0]
        angle = struct.unpack('d', bytearray(data[17:]))[0]
        
        state[0] = x + offset[0]
        state[1] = y + offset[1]
        state[2] = angle + offset[2]
        
        # Terminate program
        if (1 == data[0]):
            print("State: ", np.around(state, 3))
            print("Bumper pressed")
            exit(0)
        
        # Store the trajectory and estimated trajectory
        traj = np.concatenate((traj, [state]), axis=0)

        dist = dist_points([state[0], state[1]], [target_pos[0], target_pos[1]])
        dist_next = dist_points([state[0], state[1]], [next_target_pos[0], next_target_pos[1]])
        if ((dist < goal_dist) or (dist_next < dist)):
            return state, traj, True
            
        elif (dist > prev_dist): 
            print("To far off: ", dist)
            return state, traj, False
                        
        
        # Update the new distance
        prev_dist = dist
