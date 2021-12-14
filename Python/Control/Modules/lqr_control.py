# Import important libraries
import numpy as np
import matplotlib.pyplot as plt
from Modules.kinematics import *
import smbus
import struct

show_animation = False

bus = smbus.SMBus(1)
address = 25


def closed_loop_prediction(desired_traj):
    # Simulation Parameters
    T = desired_traj.shape[0]  # Maximum simulation time
    goal_dis = 0.01  # How close we need to get to the goal
    goal = desired_traj[-1, :]  # Coordinates of the goal
    dt = 0.1  # Timestep interval
    time = 0.0  # Starting time

    # Initial state of the car
    state = np.array([desired_traj[0, 0], desired_traj[0, 1], 0])

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
    while T >= time:
        # Point to track
        ind = int(np.floor(time))

        goal_i = desired_traj[ind, :]

        # Generate optimal control commands
        u_lqr = dLQR(DiffDrive, Q, R, state, goal_i[0:3], dt)

        # Set motor speed
        input = [int(np.uint8(u_lqr[0] * 1000)),
                 int(np.uint8(u_lqr[1] * 10000 + 128))]
        bus.write_i2c_block_data(address, 0, input)

        # Add sensors and update position
        # Move forwad in time
        data = bus.read_i2c_block_data(address, 0, 25)
        print("x: ", struct.unpack('d', bytearray(data[1:9]))[0])
        print("y: ", struct.unpack('d', bytearray(data[9:17]))[0])
        print("angle: ", struct.unpack('d', bytearray(data[17:]))[0])
        if (data[0] == 1):
            print("bumper pressed")
            bus.write_i2c_block_data(address, 0, [0, 0])
            return t, traj
        

        # Store the trajectory and estimated trajectory
        t.append(time)
        traj = np.concatenate((traj, [state]), axis=0)

        # Check to see if the robot reached goal
        if np.linalg.norm(state[0:2]-goal[0:2]) <= goal_dis:
            print("Goal reached")
            break

        #if np.linalg.norm(state[0:2]-goal_i[0:2]) <= 0.1:
        #    # Increment time
        #    time = time + dt
        time = time + dt
    
    bus.write_i2c_block_data(address, 0, [0, 0])

    return t, traj
