import smbus
import numpy as np
import atexit
import struct
import time

bus = smbus.SMBus(1)
address = 25  

#def goodbye():
#    data = [0, 0, 0]
#    bus.write_i2c_block_data(address, 0, data)
#    print("Program exited successfully!")



data = [0, 255, 0]
bus.write_i2c_block_data(address, 0, data)

while True:
    try:
        state = [0.0, 0.0, 0.0]
        data = bus.read_i2c_block_data(address, 0, 25)
        x = struct.unpack('d', bytearray(data[1:9]))[0]
        y = struct.unpack('d', bytearray(data[9:17]))[0]
        angle = struct.unpack('d', bytearray(data[17:]))[0]

        state[0] = x
        state[1] = y
        state[2] = angle
        print("State: ", np.around(state, 3))
        time.sleep(1)
    except KeyboardInterrupt:
        data = [0, 0, 0]
        bus.write_i2c_block_data(address, 0, data)
        print("Program exited successfully!")
        exit(1)


