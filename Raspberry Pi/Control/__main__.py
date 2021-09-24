from Modules.utils import compute_traj
from Modules.lqr_control import *
import subprocess
import time

def reset_encoder():
    subprocess.call("./../reset.sh")
    time.sleep(2)

if __name__ == '__main__':
    print("LQR steering control tracking start")

    # Create the track waypoints
    goal =  [5.0, 0.5]
    ax = [0, goal[0]]
    ay = [0, goal[1]]

    # Compute the desired trajectory
    desired_traj = compute_traj(ax,ay)

    reset_encoder()
    t, trajectory1 = closed_loop_prediction([goal[0], goal[1]])

    #reset_encoder()
    #t, trajectory2 = closed_loop_prediction([goal[0], 0.0])       

    #reset_encoder()
    #t, trajectory3 = closed_loop_prediction([goal[0], -goal[1]])       

    # Display the trajectory that the mobile robot executed
    plt.close()
    flg, _ = plt.subplots(1)
    plt.plot(trajectory1[:,0], trajectory1[:,1], "-g", label="Tracking +")
    #plt.plot(trajectory2[:,0], trajectory2[:,1], "-b", label="Tracking")
    #plt.plot(trajectory3[:,0], trajectory3[:,1], "-r", label="Tracking -")
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.legend()
    plt.show()
    
    # Print traveled distance
    data = bus.read_i2c_block_data(address, 0, 25)
    x = struct.unpack('d', bytearray(data[1:9]))[0]
    y = struct.unpack('d', bytearray(data[9:17]))[0]
    angle = struct.unpack('d', bytearray(data[17:]))[0]
    print("x: ", x)
    print("y: ", y)
    print("angle: ", angle)
    