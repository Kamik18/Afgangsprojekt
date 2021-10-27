import smbus
import struct


class Turtle:
    def __init__(self):
        # Create an I2C bus
        self.bus = smbus.SMBus(1)
        # Set address
        self.address = 25

    def get_pos(self):
        data = self.bus.read_i2c_block_data(self.address, 0, 25)
        bumper = data[0]
        x = struct.unpack('d', bytearray(data[1:9]))[0]
        y = struct.unpack('d', bytearray(data[9:17]))[0]
        angle = struct.unpack('d', bytearray(data[17:]))[0]

        return [x, y, angle], bumper

    def set_velocity(self, lin, ang):
        # Set motor speed
        cmd = [0, 0, 0]
        cmd[0] = int(abs(lin) * 255)
        if (ang > 0):
            cmd[1] = int(abs(ang) * 255)
        else:
            cmd[2] = int(abs(ang) * 255)
        self.bus.write_i2c_block_data(self.address, 0, cmd)

turtle = Turtle()
