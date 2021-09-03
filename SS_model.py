from defines import max_linear_velocity, max_angular_velocity
import numpy as np

def state_space_model(A, state_t_minus_1, B, control_input_t_minus_1):
    """
    Calculates the state at time t given the state at time t-1 and
    the control inputs applied at time t-1
     
    :param: A   The A state transition matrix
        3x3 NumPy Array
    :param: state_t_minus_1     The state at time t-1  
        3x1 NumPy Array given the state is [x,y,yaw angle] ---> 
        [meters, meters, radians]
    :param: B   The B state transition matrix
        3x2 NumPy Array
    :param: control_input_t_minus_1     Optimal control inputs at time t-1  
        2x1 NumPy Array given the control input vector is 
        [linear velocity of the car, angular velocity of the car]
        [meters per second, radians per second]
         
    :return: State estimate at time t
        3x1 NumPy Array given the state is [x,y,yaw angle] --->
        [meters, meters, radians]
    """
    # These next 6 lines of code which place limits on the angular and linear 
    # velocities of the robot car can be removed if you desire.
    control_input_t_minus_1[0] = np.clip(control_input_t_minus_1[0],
                                                                            -max_linear_velocity,
                                                                            max_linear_velocity)
    control_input_t_minus_1[1] = np.clip(control_input_t_minus_1[1],
                                                                            -max_angular_velocity,
                                                                            max_angular_velocity)
    state_estimate_t = (A @ state_t_minus_1) + (B @ control_input_t_minus_1) 
             
    return state_estimate_t
     