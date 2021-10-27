# Import important libraries
import numpy as np
import Modules.kinematics as kin
from Modules.Turtle import turtle


def get_R():
    return np.array([[0.1, 0],  # Linear velocity
                     [0, 0.2]])  # Angular velocity


def get_Q():
    penalty_pos = 0.5
    penalty_ang = 1.0
    return np.array([[penalty_pos, 0, 0],  # X
                     [0, penalty_pos, 0],  # Y
                     [0, 0, penalty_ang]]) # Theta


def validate(value, limit_lower, limit_upper):
    result = value
    if value > limit_upper:
        result = limit_upper
    elif value < limit_lower:
        result = limit_lower
    return result


def dist_points(point1, point2):
    return np.linalg.norm(point1 - point2)


def closed_loop_prediction(desired_traj, simulation=False):
    # Simulation Parameters
    goal_dis = 0.01  # How close we need to get to the goal
    dist_threshold = 0.01
    dt = 1 / 20  # Timestep interval
    VELOCITY = 0.8

    # Initial States
    # Initial state of the car
    state = np.array(
        [desired_traj[0, 0], desired_traj[0, 1], desired_traj[0, 2]])

    OFFSET = state[0:2]

    if (not simulation):
        state, _ = turtle.get_pos(OFFSET)

    Q = get_Q()
    R = get_R()

    # Create objects for storing states and estimated state
    trajectory = np.array([state])

    try:
        prev_distance = np.inf
        index = 1

        while (index < len(desired_traj)):
            # Generate optimal control commands
            u_lqr = kin.dLQR(Q, R, state, desired_traj[index, 0:3], dt)

            if (index < (len(desired_traj) - 0)):
                factor = VELOCITY / (abs(u_lqr[0]) + abs(u_lqr[1]))
                u_lqr *= validate(factor, 0, np.inf)

            # Add sensors and update position
            # Move forwad in time
            if (simulation):
                state = kin.forward(state, u_lqr, dt)
            else:
                turtle.set_velocity(u_lqr[0], u_lqr[1])
                state, bumper = turtle.get_pos(OFFSET)

                if (bumper == 1):
                    print("Bumper pressed")
                    break

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
            if (index < (len(desired_traj) - 1)):
                if ((distance_target < dist_threshold) or
                    (distance_target >= prev_distance) or
                        (distance_target > distance_target_next)):
                    prev_distance = distance_target_next
                    index += 1
                    print(index, " of ", len(desired_traj))

            # Validate if goal is reached
            elif ((distance_target < goal_dis) or
                  (distance_target >= prev_distance)):
                print("Completed run, offset: ", dist_points(
                    state[0:2], desired_traj[-1, 0:2]))
                break

            # Update the distance error
            else:
                prev_distance = distance_target

    except KeyboardInterrupt:
        print("Interrupted")

    # Return the trajectory
    return trajectory
