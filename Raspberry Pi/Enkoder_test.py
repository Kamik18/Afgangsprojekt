import smbus
import time
import struct
import subprocess
import numpy as np

def reset_encoder():
    subprocess.call("./reset.sh")
    time.sleep(2)

# Reset the encoders
reset_encoder()
    
# Create I2C instance
bus = smbus.SMBus(1)
address = 25

# Wait for I2C module to be ready
time.sleep(1)

while True:
    state = [0.0, 0.0, 0.0]
    data = bus.read_i2c_block_data(address, 0, 25)
    x = struct.unpack('d', bytearray(data[1:9]))[0]
    y = struct.unpack('d', bytearray(data[9:17]))[0]
    angle = struct.unpack('d', bytearray(data[17:]))[0]
    
    # Terminate program
    if (1 == data[0]):
        print("Bumper pressed")
        exit(0)
    
    state[0] = x 
    state[1] = y 
    state[2] = angle 
    print("State: ", np.around(state, 3))

    time.sleep(1)