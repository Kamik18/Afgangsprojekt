"""
Implementation of the two-wheeled differential drive robot car
and its controller.

Our goal in using LQR is to find the optimal control inputs:
  [linear velocity of the car, angular velocity of the car]

We want to both minimize the error between the current state
and a desired state, while minimizing actuator effort
(e.g. wheel rotation rate). These are competing objectives because a
large u (i.e. wheel rotation rates) expends a lot of
actuator energy but can drive the state error to 0 really fast.
LQR helps us balance these competing objectives.

If a system is linear, LQR gives the optimal control inputs that
takes a system's state to 0, where the state is
"current state - desired state".
"""
# Import required libraries
import numpy as np
import scipy.linalg as la


def forward(x0, u, dt=0.1):
    """
    Computes the forward kinematics for the system.

    Input
      :param x0: The starting state (position) of the system (units:[m,m,rad])
              np.array with shape (3,) ->
              (X, Y, THETA)
      :param u:  The control input to the system
              2x1 NumPy Array given the control input vector is
              [linear velocity of the car, angular velocity of the car]
              [meters per second, radians per second]
      :param dt: Change in time (units: [s])

    Output
      :return: x1: The new state of the system (X, Y, THETA)
    """
    # Control input
    u_linvel = u[0]
    u_angvel = u[1]

    # Velocity in the x and y direction in m/s
    x_dot = u_linvel * np.cos(x0[2])
    y_dot = u_linvel * np.sin(x0[2])

    # Calculate the new state of the system
    return np.array([x0[0] + x_dot * dt,  # X
                     x0[1] + y_dot * dt,  # Y
                     x0[2] + u_angvel * dt])  # THETA


def linearize(angle, dt=0.1):
    """
    Creates a linearized version of the dynamics of the differential
    drive robotic system (i.e. a
    robotic car where each wheel is controlled separately.

    The system's forward kinematics are nonlinear due to the sines and
    cosines, so we need to linearize
    it by taking the Jacobian of the forward kinematics equations with respect
     to the control inputs.

    Our goal is to have a discrete time system of the following form:
    x_t+1 = Ax_t + Bu_t where:

    Input
      :param x: The state of the system (units:[m,m,rad]) ->
                np.array with shape (3,) ->
                (X, Y, THETA) ->
                X_system = [x1, x2, x3]
      :param dt: The change in time from time step t to time step t+1

    Output
      :return: A: Matrix A is a 3x3 matrix (because there are 3 states) that
                  describes how the state of the system changes from t to t+1
                  when no control command is executed. Typically,
                  a robotic car only drives when the wheels are turning.
                  Therefore, in this case, A is the identity matrix.
      :return: B: Matrix B is a 3 x 2 matrix (because there are 3 states and
                  2 control inputs) that describes how
                  the state (X, Y, and THETA) changes from t to t + 1 due to
                  the control command u.
                  Matrix B is found by taking the The Jacobian of the three
                  forward kinematics equations (for X, Y, THETA)
                  with respect to u (3 x 2 matrix)
    """
    ####### A Matrix #######
    # A matrix is the identity matrix
    A = np.array([[1.0,   0,  0],
                  [0, 1.0,  0],
                  [0,   0, 1.0]])

    ####### B Matrix #######
    B = np.array([[np.cos(angle)*dt, 0],
                  [np.sin(angle)*dt, 0],
                  [0, dt]])

    return A, B


def pi_2_pi(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi


def dLQR(Q, R, x, xf, dt=0.1):
    """
    Discrete-time linear quadratic regulator for a non-linear system.

    Compute the optimal control given a nonlinear system, cost matrices,
    a current state, and a final state.
    Compute the control variables that minimize the cumulative cost.

    Solve for P using the dynamic programming method.

    Assume that Qf = Q

    Input:
      :param F: The dynamics class object (has forward and linearize functions
                implemented)
      :param Q: The state cost matrix Q -> np.array with shape (3,3)
      :param R: The input cost matrix R -> np.array with shape (2,2)
      :param x: The current state of the system x -> np.array with shape (3,)
      :param xf: The desired state of the system xf -> np.array with shape (3,)
      :param dt: The size of the timestep -> float

    Output
      :return: u_t_star: Optimal action u for the current state
                   [linear velocity of the car, angular velocity of the car]
                   [meters per second, radians per second]

    __matmul__: np.matmul(A, B) = A @ B
    """
    # We want the system to stabilize at xf,
    # so we let x - xf be the state.
    # Actual state - desired state
    x_error = x - xf

    # Ensure the anglehasn't rotated 360 degrees
    x_error = [x_error[0], x_error[1], pi_2_pi(x_error[2])]

    # Calculate the A and B matrices
    A, B = linearize(x[2], dt)

    # Solutions to discrete LQR problems are obtained using dynamic
    # programming.
    # The optimal solution is obtained recursively, starting at the last
    # time step and working backwards.
    N = 50

    # Create a list of N + 1 elements
    P = [None] * (N + 1)

    # Assume Qf = Q
    Qf = Q

    # 1. LQR via Dynamic Programming
    P[N] = Qf

    # 2. For t = N, ..., 1
    for t in range(N, 0, -1):
        # Discrete-time Algebraic Riccati equation to calculate the optimal
        # state cost matrix
        P[t-1] = Q + A.T @ P[t] @ A - (A.T @ P[t] @ B) @ la.pinv(
            R + B.T @ P[t] @ B) @ (B.T @ P[t] @ A)

    # Create a list of N elements
    K = [None] * N
    u = [None] * N

    # 3 and 4. For t = 0, ..., N - 1
    for t in range(N):
        # Calculate the optimal feedback gain K_t
        K[t] = -la.pinv(R + B.T @ P[t+1] @ B) @ B.T @ P[t+1] @ A

    for t in range(N):
        # Calculate the optimal control input
        u[t] = K[t] @ x_error

    # Optimal control input is u_t_star
    u_t_star = u[N-1]

    # Return the optimal control inputs
    return u_t_star
