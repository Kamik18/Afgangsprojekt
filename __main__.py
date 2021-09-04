from Modules.utils import compute_traj
from Modules.lqr_control import *

if __name__ == '__main__':
    print("LQR steering control tracking start")
 
    # Create the track waypoints
    ax = [8.3, 8.0, 7.2, 6.2, 6.5, 1.5,-2.0,-3.5, 2, 10]
    ay = [0.7, 4.3, 4.5, 4.0, 0.7, 1.3, 3.3, 1.5, 2, 10]
    
    # Compute the desired trajectory
    desired_traj = compute_traj(ax,ay)
 
    t, trajectory = closed_loop_prediction(desired_traj)
 
    # Display the trajectory that the mobile robot executed
    if show_animation:
        plt.close()
        flg, _ = plt.subplots(1)
        plt.plot(ax, ay, "xb", label="Inputs")
        plt.plot(desired_traj[:,0], desired_traj[:,1], "-r", label="Trajectory")
        plt.plot(trajectory[:,0], trajectory[:,1], "-g", label="Tracking")
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.legend()
 
        plt.show()
    