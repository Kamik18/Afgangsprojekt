"""
A robot will follow a racetrack using an LQR controller to estimate 
the state (i.e. x position, y position, yaw angle) at each timestep
"""

# Import important libraries
import numpy as np
import matplotlib.pyplot as plt
from Modules.kinematics import *
import struct
import math

show_animation = True


def dist_points(point1, point2):
    return np.linalg.norm(point1 - point2)


def closed_loop_prediction(desired_traj):
    # Simulation Parameters
    T = desired_traj.shape[0]  # Maximum simulation time
    goal_dis = 0.01  # How close we need to get to the goal
    dist_threshold = 0.10
    dt = 1 / 20  # Timestep interval

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
    trajectory = np.array([state])

    prev_distance = np.inf
    index = 1
    while (index < (len(desired_traj) - 1)):
        # Generate optimal control commands
        u_lqr = dLQR(DiffDrive, Q, R, state, desired_traj[index, 0:3], dt)

        # Add sensors and update position
        # Move forwad in time
        state = DiffDrive.forward(state, u_lqr, dt)

        # Store the trajectory and estimated trajectory
        trajectory = np.concatenate((trajectory, [state]), axis=0)

        # Check to see if the robot reached goal
        if dist_points(state[0:2], desired_traj[-1, 0:2]) <= goal_dis:
            print("Goal reached")
            break

        distance_target = dist_points(state[0:2], desired_traj[index, 0:2])
        distance_target_next = dist_points(
            state[0:2], desired_traj[index + 1, 0:2])
        if ((distance_target < dist_threshold) or
            (distance_target >= prev_distance) or
                (distance_target > distance_target_next)):
            index += 1
            prev_distance = distance_target_next
        else:
            prev_distance = distance_target

        # Plot the vehicles trajectory
        if show_animation:
            plt.cla()
            plt.plot(desired_traj[:, 0],
                     desired_traj[:, 1], ".r", label="course")
            plt.plot(trajectory[:, 0], trajectory[:, 1],
                     "-b", label="trajectory")
            plt.plot(desired_traj[index, 0],
                     desired_traj[index, 1], "xb", label="target")
            plt.legend()
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.0001)

    # Return the trajectory
    return trajectory
