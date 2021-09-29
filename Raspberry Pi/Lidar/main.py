import atexit
import serial
from Modules import hokuyo
from Modules import serial_port
import matplotlib.pyplot as plt
import math
import time
import os
import numpy as np

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


uart_port = '/dev/ttyACM0'
uart_speed = 19200

if __name__ == '__main__':
    laser_serial = serial.Serial(port=uart_port, baudrate=uart_speed, timeout=0.5)
    port = serial_port.SerialPort(laser_serial)

    laser = hokuyo.Hokuyo(port)

    print('laser_on')
    print(laser.laser_on())
    print(laser.set_high_sensitive(True))
    print(laser.set_motor_speed())
    
    
    print("Fetch data")
    t0 = time.process_time()
    for i in range(100):
        data = laser.get_single_scan()
    time_elapsed = time.process_time() - t0
    print("Elapsed time: ", time_elapsed, " frequency: ", 1/(time_elapsed / (i+1)))
    
    
    data = laser.get_single_scan()
    x = []
    y = []
    for item in data:
        length = data[item]
        if length > 5:
            angle = -math.radians(item)
            x.append(length * np.cos(angle))
            y.append(length * np.sin(angle))
    
    limits_x = [0, 100]
    limits_y = [0, 0]
    
    plt.ion()
    figure, ax = plt.subplots(figsize=(10,8))
    line_dir, = ax.plot(limits_x, limits_y)
    line1, = ax.plot(x,y, '.')
    plt.title("test", fontsize=20)
    
    try:
        while (True):
            data = laser.get_single_scan()
                
            new_x = []
            new_y = []
            for item in data:
                length = data[item]
                if length > 25:
                    angle = -math.radians(item )
                    new_x.append(length * np.cos(angle))
                    new_y.append(length * np.sin(angle))
                    
            line1.set_xdata(new_x)
            line1.set_ydata(new_y)
            
            figure.canvas.draw()
            figure.canvas.flush_events()
            
    except KeyboardInterrupt:
        print("Interrupted")
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
