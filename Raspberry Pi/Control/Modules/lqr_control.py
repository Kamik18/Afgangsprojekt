"""
A robot will follow a racetrack using an LQR controller to estimate 
the state (i.e. x position, y position, yaw angle) at each timestep
"""

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

    # Initial States
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

        # Generate optimal control commands
        u_lqr = dLQR(DiffDrive, Q, R, state, goal[0:3], dt)
        print("u_lqr: ", u_lqr)

        # Set motor speed
        input = [int(np.uint8(u_lqr[0] * 1000)),
                 int(np.uint8(u_lqr[1] * 10000 + 128))]
        bus.write_i2c_block_data(address, 0, input)

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
        print("State: ", state)
        
        if (data[0] == 1) or (state[0] > goal[0]):
            print("bumper pressed")
            bus.write_i2c_block_data(address, 0, [0, 128])
            return t, traj
        

        # Store the trajectory and estimated trajectory
        t.append(time)
        traj = np.concatenate((traj, [state]), axis=0)

        # Check to see if the robot reached goal
        if np.linalg.norm(state[0:2]-goal[0:2]) <= goal_dis:
            print("Goal reached")
            break

        time = time + dt
        
        '''
        # Plot the vehicles trajectory
        if time % 1 < 0.1 and show_animation:
            plt.cla()
            plt.plot(desired_traj[:, 0],
                     desired_traj[:, 1], "-r", label="course")
            plt.plot(traj[:, 0], traj[:, 1], "ob", label="trajectory")
            plt.legend()
            plt.axis("equal")
            plt.grid(True)
            plt.title("speed[m/s]:" + str(round(np.mean(u_lqr), 2)) +
                      ",target index:" + str(ind))
            plt.pause(0.0001)
        '''
    
    bus.write_i2c_block_data(address, 0, [0, 128])
    
    return t, traj
