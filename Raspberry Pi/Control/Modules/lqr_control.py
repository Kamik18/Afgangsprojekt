"""
A robot will follow a racetrack using an LQR controller to estimate 
the state (i.e. x position, y position, yaw angle) at each timestep
"""

# Import important libraries
import numpy as np
import matplotlib.pyplot as plt
from Modules.kinematics import *
import math
import smbus
import struct

show_animation = False

bus = smbus.SMBus(1)
address = 25

def constrain(val, min_val, max_val):
    if val < min_val: return min_val
    if val > max_val: return max_val
    return val

def closed_loop_prediction(target_pos):
    # Simulation Parameters
    goal_dist = 0.01  # How close we need to get to the goal
    goal =  np.array([target_pos[0], target_pos[1], math.atan2(target_pos[1], target_pos[0])])
    print("goal: ", goal)

    dt = 0.05  # Timestep interval
    time = 0.0  # Starting time

    # Initial States
    # Initial state of the car
    state = np.array([target_pos[0], target_pos[1], 0])

    # Get the Cost-to-go and input cost matrices for LQR
    Q = get_Q()  # Defined in kinematics.py
    R = get_R()  # Defined in kinematics.py

    # Initialize the Car and the Car's landmark sensor
    DiffDrive = DifferentialDrive()

    # Process noise and sensor measurement noise
    V = DiffDrive.get_V()

    # Create objects for storing states and estimated state
    t = [time]
    traj = np.array([state])

    ind = 0
    while True:
        # Point to track
        ind = int(np.floor(time))

        # Generate optimal control commands
        u_lqr = dLQR(DiffDrive, Q, R, state, goal[0:3], dt)

        # Set motor speed
        cmd = [0, 0, 0]
        if (u_lqr[1] > 0):
            cmd[0] = constrain(int(abs(u_lqr[0]) * 100), 0, 255)
            cmd[1] = constrain(int(abs(u_lqr[1]) * 1000), 0, 255)
        else:
            cmd[0] = constrain(int(abs(u_lqr[0]) * 100), 0, 255)
            cmd[2] = constrain(int(abs(u_lqr[1]) * 1000), 0, 255)
        print(cmd)
        bus.write_i2c_block_data(address, 0, cmd)

        # Add sensors and update position
        # Move forwad in time
        state = [0.0, 0.0, 0.0]
        data = bus.read_i2c_block_data(address, 0, 25)
        x = struct.unpack('d', bytearray(data[1:9]))[0]
        y = struct.unpack('d', bytearray(data[9:17]))[0]
        angle = struct.unpack('d', bytearray(data[17:]))[0]
        
        state[0] = x
        state[1] = y
        state[2] = angle
        print("State: ", np.around(state, 3))
        
        # Store the trajectory and estimated trajectory
        t.append(time)
        traj = np.concatenate((traj, [state]), axis=0)


        if (data[0] == 1) or (state[0] > (goal[0] + goal_dist)):
            print("goal: ", goal)
            bus.write_i2c_block_data(address, 0, [0, 0, 0])
            return t, traj
            
        time = time + dt
        
    bus.write_i2c_block_data(address, 0, [0, 0, 0])
    
    return t, traj
