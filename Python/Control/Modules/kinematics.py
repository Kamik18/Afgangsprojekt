# Import required libraries
import numpy as np
import scipy.linalg as la


class DifferentialDrive(object):
    def __init__(self):
        """
        Initializes the class
        """

    def linearize(self, x, dt=0.1):
        THETA = x[2]

        ####### A Matrix #######
        # A matrix is the identity matrix
        A = np.array([[1.0,   0,  0],
                      [0, 1.0,  0],
                      [0,   0, 1.0]])

        ####### B Matrix #######
        B = np.array([[np.cos(THETA)*dt, 0],
                      [np.sin(THETA)*dt, 0],
                      [0, dt]])

        return A, B


def dLQR(F, Q, R, x, xf, dt=0.1):
    # We want the system to stabilize at xf,
    # so we let x - xf be the state.
    # Actual state - desired state
    x_error = x - xf

    # Calculate the A and B matrices
    A, B = F.linearize(x, dt)

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


def get_R():
    R = np.array([[0.01, 0],  # Penalization for linear velocity effort
                  [0, 0.01]])  # Penalization for angular velocity effort

    return R


def get_Q():
    Q = np.array([[0.4, 0, 0],  # Penalize X position error (global coordinates)
                  # Penalize Y position error (global coordinates)
                  [0, 0.4, 0],
                  [0, 0, 0.4]])  # Penalize heading error (global coordinates)

    return Q
