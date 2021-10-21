# Import important libraries
import numpy as np
import matplotlib.pyplot as plt
import Modules.kinematics as kin
import struct
import math

show_animation = True

def get_R():
    """
    This function provides the R matrix to the lqr_control simulator.

    Returns the input cost matrix R.

    Experiment with different gains.
    This matrix penalizes actuator effort 
    (i.e. rotation of the motors on the wheels).
    The R matrix has the same number of rows as are actuator states 
    [linear velocity of the car, angular velocity of the car]
    [meters per second, radians per second]
    This matrix often has positive values along the diagonal.
    We can target actuator states where we want low actuator 
    effort by making the corresponding value of R large.   

    Output
      :return: R: Input cost matrix
    """
    R = np.array([[0.01, 0],  # Penalization for linear velocity effort
                  [0, 0.02]])  # Penalization for angular velocity effort

    return R


def get_Q():
    """
    This function provides the Q matrix to the lqr_control simulator.

    Returns the state cost matrix Q.

    Experiment with different gains to see their effect on the vehicle's 
    behavior.
    Q helps us weight the relative importance of each state in the state 
    vector (X, Y, THETA). 
    Q is a square matrix that has the same number of rows as there are states.
    Q penalizes bad performance.
    Q has positive values along the diagonal and zeros elsewhere.
    Q enables us to target states where we want low error by making the 
    corresponding value of Q large.
    We can start with the identity matrix and tweak the values through trial 
    and error.

    Output
      :return: Q: State cost matrix (3x3 matrix because the state vector is 
                  (X, Y, THETA))
    """
    Q = np.array([[0.4, 0, 0],  # Penalize X position error (global coordinates)
                  # Penalize Y position error (global coordinates)
                  [0, 0.4, 0],
                  [0, 0, 0.85]])  # Penalize heading error (global coordinates)

    return Q

def validate(value, limit_lower, limit_upper):
    result = value
    if value > limit_upper:
        result = limit_upper
    elif value < limit_lower:
        result = limit_lower
    return result


def dist_points(point1, point2):
    return np.linalg.norm(point1 - point2)


def closed_loop_prediction(desired_traj):
    # Simulation Parameters
    T = desired_traj.shape[0]  # Maximum simulation time
    goal_dis = 0.01  # How close we need to get to the goal
    dist_threshold = 0.05
    dt = 1 / 20  # Timestep interval
    VELOCITY = 0.8

    # Initial States
    # Initial state of the car
    state = np.array(
        [desired_traj[0, 0], desired_traj[0, 1], desired_traj[0, 2]])
    print("Initial state ", state)

    # Get the Cost-to-go and input cost matrices for LQR
    Q = get_Q()  # Defined in kinematics.py
    R = get_R()  # Defined in kinematics.py

    # Create objects for storing states and estimated state
    trajectory = np.array([state])

    prev_distance = np.inf
    index = 1
    while (index < len(desired_traj)):
        # Generate optimal control commands
        u_lqr = kin.dLQR(Q, R, state, desired_traj[index, 0:3], dt)

        if (index < (len(desired_traj) - 1)):
            factor = VELOCITY / (abs(u_lqr[0]) + abs(u_lqr[1]))
            u_lqr *= validate(factor, 0, np.inf)

        # Add sensors and update position
        # Move forwad in time
        state = kin.forward(state, u_lqr, dt)

        # Store the trajectory and estimated trajectory
        trajectory = np.concatenate((trajectory, [state]), axis=0)

        # Calculate the distance
        distance_target = dist_points(state[0:2], desired_traj[index, 0:2])

        # Validate if the target is the final target
        if (index < (len(desired_traj) - 1)):
            distance_target_next = dist_points(
                state[0:2], desired_traj[index + 1, 0:2])
        else:
            distance_target_next = distance_target

        # Validate when to shift target
        if (((distance_target < dist_threshold) and (index < (len(desired_traj) - 1))) or
            (distance_target >= prev_distance) or
                (distance_target > distance_target_next)):
            if (index < (len(desired_traj) - 1)):
                prev_distance = distance_target_next
                index += 1
            else:
                print("Completed run, offset: ", dist_points(
                    state[0:2], desired_traj[-1, 0:2]))
                break

        else:
            prev_distance = distance_target

        # Plot the vehicles trajectory
        if show_animation:
            plt.cla()
            plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [
                exit(0) if event.key == 'escape' else None])
            plt.plot(trajectory[:, 0], trajectory[:, 1],
                     "-g", label="Tracking")
            plt.plot(desired_traj[index, 0],
                     desired_traj[index, 1], "xb", label="Target")
            plt.title(str(np.around(u_lqr, 2)) + " m/s")
            plt.legend()
            plt.grid(True)
            plt.axis("equal")
            plt.xlabel("x[m]")
            plt.ylabel("y[m]")
            plt.pause(0.0001)

    # Return the trajectory
    return trajectory
