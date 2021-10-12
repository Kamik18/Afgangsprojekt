import atexit
import serial
from Modules import hokuyo
from Modules import serial_port
import matplotlib.pyplot as plt
import numpy as np

# UART - Lidar communication
uart_port = '/dev/ttyACM0'
uart_speed = 19200

def goodbye():
    print('get_version_info')
    print(laser.get_version_info())
    print('get_sensor_specs')
    print(laser.get_sensor_specs())
    print('reset')
    print(laser.reset())
    print('laser_off')
    print(laser.laser_off())
    print('Complete')
    exit(0)

def AD2pos(distance, angle, robot_position):
    x = distance * np.cos(angle - robot_position[2]) + robot_position[0]
    y = -distance * np.sin(angle - robot_position[2]) + robot_position[1]
    return (int(x), int(y))

def convert_data_set(data, robot_position=(0, 0, 0)):
    points = []
    if not data:
        pass
    else:
        for point in data:
            length = data[point]
            if length > 20:
                coordinates = AD2pos(length, np.radians(point), robot_position)
                points.append(coordinates)
    return points


if __name__ == '__main__':
    laser_serial = serial.Serial(
        port=uart_port, baudrate=uart_speed, timeout=0.5)
    port = serial_port.SerialPort(laser_serial)

    laser = hokuyo.Hokuyo(port)

    print('laser_on')
    print(laser.laser_on())
    print(laser.set_high_sensitive(True))
    print(laser.set_motor_speed())

    print("Fetch data")
    data_set = convert_data_set(laser.get_single_scan())

    # Plot lines
    plt.cla()
    if data_set:
        x, y = zip(*data_set)
        plt.plot(x, y, '-b')
    #plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()

    print(data_set)

    # Terminate session
    goodbye()    
