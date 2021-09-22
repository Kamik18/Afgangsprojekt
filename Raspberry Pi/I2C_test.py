import smbus
import time

# Create I2C instance
bus = smbus.SMBus(1)
address = 25

# Wait for I2C module to be ready
time.sleep(1)

# Data package
left_pwm = 40  # 0 - 255
left_dir = 0    # 0 - 1
right_pwm = 40 # 0 - 255
right_dir = 0   # 0 - 1

# Write data
data = [left_pwm, left_dir, right_pwm, right_dir]
bus.write_i2c_block_data(address, 0, data)

# Read 1 byte: 1 collision occured / 0 fine
recv = bus.read_i2c_block_data(address, 0, 1)
print(recv)
