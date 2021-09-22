from Modules.utils import compute_traj
from Modules.lqr_control import *
import subprocess
import time

if __name__ == '__main__':
    print("LQR steering control tracking start")

    subprocess.call("./../reset.sh")
    time.sleep(1)
 
    # Create the track waypoints
    ax = [0, 0.1, 5]
    ay = [1, 1 , 0]
    
    # Compute the desired trajectory
    desired_traj = compute_traj(ax,ay)
 
    t, trajectory = closed_loop_prediction(desired_traj)
 
    # Display the trajectory that the mobile robot executed
    plt.close()
    flg, _ = plt.subplots(1)
    plt.plot(desired_traj[:,0], desired_traj[:,1], "-r", label="Trajectory")
    plt.plot(trajectory[:,0], trajectory[:,1], "-g", label="Tracking")
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.legend()
    plt.show()
    