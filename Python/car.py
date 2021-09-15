from pyPS4Controller.controller import Controller
import RPi.GPIO as GPIO
import smbus
import math
import atexit
import time

def goodbye():
    print("Program exited successfully!")
    #GPIO.cleanup()                              # resets GPIO ports used back to input mode  

bus = smbus.SMBus(1)
address = 25  


# Wait for I2C module to be ready
time.sleep(1)



class MyCtr(Controller):

    camera = False 
    max_value = 32768
    prev_value = 0

    # Data package
    left_pwm = 40   # 0 - 255
    left_wheel = 0  # 0 (forward) - 1 (reverse)
    right_pwm = 40  # 0 - 255
    right_wheel = 0 # 0 (reverse) - 1 (forward)

    drive_forward = False
    drive_reverse = False
    turn_left = False
    turn_right = False

    def __init__(self, **kwargs):
        Controller.__init__(self, **kwargs)
    
    def get_motor_value_analog(value):
        if value is 0:
            return 0
        else:
            new_value = math.trunc((abs(value) / MyCtr.max_value) * 255)
            if new_value > MyCtr.prev_value:
                MyCtr.prev_value = MyCtr.prev_value + 1
            else:
                MyCtr.prev_value = new_value
            return MyCtr.prev_value

    def get_motor_value_buttons(value):
        if value is 0:
            return 0
        else:
            new_value = math.trunc(((value+MyCtr.max_value) / (MyCtr.max_value*2)) * 255)
            if new_value > MyCtr.prev_value:
                MyCtr.prev_value = MyCtr.prev_value + 1
            else:
                MyCtr.prev_value = new_value
            return MyCtr.prev_value

    def set_left_pwm(value):
        MyCtr.left_pwm = MyCtr.get_motor_value_analog(value)

    def set_right_pwm(value):
        MyCtr.right_pwm = MyCtr.get_motor_value_analog(value)

    # Turn on and off camera
    def on_x_press(self):
        if MyCtr.camera is True:
            print("Turn off camera")
            MyCtr.camera = False
        else:
            print("Turn on Camera")
            MyCtr.camera = True

    def on_R2_press(self, value):
        MyCtr.drive_forward = True
        MyCtr.drive_reverse = False
        if MyCtr.turn_left:
            MyCtr.right_pwm = MyCtr.get_motor_value_buttons(value)
            if (MyCtr.left_pwm > MyCtr.right_pwm) and MyCtr.right_pwm is not 0:
                if MyCtr.left_pwm > 200:
                    MyCtr.left_pwm = 0
                else:
                    MyCtr.left_pwm = math.trunc((MyCtr.left_pwm / 255) * MyCtr.right_pwm)
        elif MyCtr.turn_right:
            MyCtr.left_pwm = MyCtr.get_motor_value_buttons(value)
            if (MyCtr.right_pwm > MyCtr.left_pwm) and MyCtr.left_pwm is not 0:
                if MyCtr.left_pwm > 200:
                    MyCtr.left_pwm = 0
                else:
                    MyCtr.right_pwm = math.trunc((MyCtr.right_pwm / 255 ) * MyCtr.left_pwm)
        else:
            MyCtr.left_pwm = MyCtr.right_pwm = MyCtr.get_motor_value_buttons(value)
        
        MyCtr.left_wheel = 0
        MyCtr.right_wheel = 1
        data = [MyCtr.left_pwm, MyCtr.left_wheel, MyCtr.right_pwm, MyCtr.right_wheel]
        bus.write_i2c_block_data(address, 0, data)

    def on_L2_press(self, value):
        MyCtr.drive_forward = False
        MyCtr.drive_reverse = True
        if MyCtr.turn_left:
            MyCtr.right_pwm = MyCtr.get_motor_value_buttons(value)
            if (MyCtr.left_pwm > MyCtr.right_pwm) and MyCtr.right_pwm is not 0:
                if MyCtr.left_pwm > 200:
                    MyCtr.left_pwm = 0
                else:
                    MyCtr.left_pwm = math.trunc((MyCtr.left_pwm / 255) * MyCtr.right_pwm)
        elif MyCtr.turn_right:
            MyCtr.left_pwm = MyCtr.get_motor_value_buttons(value)
            if (MyCtr.right_pwm > MyCtr.left_pwm) and MyCtr.left_pwm is not 0:
                if MyCtr.left_pwm > 200:
                    MyCtr.left_pwm = 0
                else:
                    MyCtr.right_pwm = math.trunc((MyCtr.right_pwm / 255 ) * MyCtr.left_pwm)
        else:
            MyCtr.left_pwm = MyCtr.right_pwm = MyCtr.get_motor_value_buttons(value)
        
        MyCtr.left_wheel = 1
        MyCtr.right_wheel = 0
        data = [MyCtr.left_pwm, MyCtr.left_wheel, MyCtr.right_pwm, MyCtr.right_wheel]
        bus.write_i2c_block_data(address, 0, data)

    def on_R2_release(self):
        MyCtr.drive_forward = False
        MyCtr.left_pwm = MyCtr.right_pwm = 0
        data = [MyCtr.left_pwm, MyCtr.left_wheel, MyCtr.right_pwm, MyCtr.right_wheel]
        bus.write_i2c_block_data(address, 0, data)
    
    def on_L2_release(self):
        MyCtr.drive_reverse = False
        MyCtr.left_pwm = MyCtr.right_pwm = 0
        data = [MyCtr.left_pwm, MyCtr.left_wheel, MyCtr.right_pwm, MyCtr.right_wheel]
        bus.write_i2c_block_data(address, 0, data)

    def on_R3_up(self, value):
        MyCtr.drive_forward = True
        MyCtr.drive_reverse = False
        if MyCtr.turn_left:
            print("turn left")
            MyCtr.right_pwm = MyCtr.get_motor_value_buttons(value)
            if (MyCtr.left_pwm > MyCtr.right_pwm) and MyCtr.right_pwm is not 0:
                if MyCtr.left_pwm > 200:
                    MyCtr.left_pwm = 0
                else:
                    MyCtr.left_pwm = math.trunc((MyCtr.left_pwm / 255) * MyCtr.right_pwm)
        elif MyCtr.turn_right:
            print("turn right")
            MyCtr.left_pwm = MyCtr.get_motor_value_buttons(value)
            if (MyCtr.right_pwm > MyCtr.left_pwm) and MyCtr.left_pwm is not 0:
                print("Too large")
                if MyCtr.left_pwm > 200:
                    MyCtr.left_pwm = 0
                else:
                    MyCtr.right_pwm = math.trunc((MyCtr.right_pwm / 255 ) * MyCtr.left_pwm)
        else:
            print("not turning")
            MyCtr.left_pwm = MyCtr.right_pwm = MyCtr.get_motor_value_buttons(value)
        
        MyCtr.left_wheel = 0
        MyCtr.right_wheel = 1
        data = [MyCtr.left_pwm, MyCtr.left_wheel, MyCtr.right_pwm, MyCtr.right_wheel]
        bus.write_i2c_block_data(address, 0, data)

    def on_R3_down(self, value):
        MyCtr.left_pwm = MyCtr.right_pwm = MyCtr.get_motor_value_analog(value)
        MyCtr.left_wheel = 1
        MyCtr.right_wheel = 0
        data = [MyCtr.left_pwm, MyCtr.left_wheel, MyCtr.right_pwm, MyCtr.right_wheel]
        bus.write_i2c_block_data(address, 0, data)

    def on_R3_y_at_rest(self):
        MyCtr.drive_forward = False
        MyCtr.drive_reverse = False
        if MyCtr.turn_right == MyCtr.turn_left:
            MyCtr.left_pwm = MyCtr.right_pwm = 0
            data = [MyCtr.left_pwm, MyCtr.left_wheel, MyCtr.right_pwm, MyCtr.right_wheel]
            bus.write_i2c_block_data(address, 0, data)

    def on_R3_x_at_rest(self):
        MyCtr.turn_left = False
        MyCtr.turn_right = False
        if MyCtr.drive_forward == MyCtr.drive_reverse:
            MyCtr.left_pwm = MyCtr.right_pwm = 0
            data = [MyCtr.left_pwm, MyCtr.left_wheel, MyCtr.right_pwm, MyCtr.right_wheel]
            bus.write_i2c_block_data(address, 0, data)

    # Turn car
    def on_R3_left(self, value):
        MyCtr.turn_left = True
        MyCtr.turn_right = False
        #print("drive_forward: ", MyCtr.drive_forward, ", drive_reverse: ", MyCtr.drive_reverse)
        if not (MyCtr.drive_forward == MyCtr.drive_reverse):
            print("left_pwm_before: ", MyCtr.left_pwm)
            MyCtr.set_left_pwm(value)
            print("left_pwm_after:  ", MyCtr.left_pwm)
        else:
            MyCtr.left_pwm = MyCtr.right_pwm = MyCtr.get_motor_value_analog(value)
            MyCtr.left_wheel = MyCtr.right_wheel= 1
            data = [MyCtr.left_pwm, MyCtr.left_wheel, MyCtr.right_pwm, MyCtr.right_wheel]
            bus.write_i2c_block_data(address, 0, data)

    def on_R3_right(self, value):
        MyCtr.turn_left = False
        MyCtr.turn_right = True
        #print("drive_forward: ", MyCtr.drive_forward, ", drive_reverse: ", MyCtr.drive_reverse)
        if not (MyCtr.drive_forward == MyCtr.drive_reverse):
            MyCtr.set_right_pwm(value)
        else:
            MyCtr.left_pwm = MyCtr.right_pwm = MyCtr.get_motor_value_analog(value)
            MyCtr.left_wheel = MyCtr.right_wheel = 0
            data = [MyCtr.left_pwm, MyCtr.left_wheel, MyCtr.right_pwm, MyCtr.right_wheel]
            bus.write_i2c_block_data(address, 0, data)
            
    def on_L3_left(self, value):
        MyCtr.turn_left = True
        MyCtr.turn_right = False
        #print("drive_forward: ", MyCtr.drive_forward, ", drive_reverse: ", MyCtr.drive_reverse)
        if not (MyCtr.drive_forward == MyCtr.drive_reverse):
            print("left_pwm_before: ", MyCtr.left_pwm)
            MyCtr.set_left_pwm(value)
            print("left_pwm_after:  ", MyCtr.left_pwm)
        else:
            MyCtr.left_pwm = MyCtr.right_pwm = MyCtr.get_motor_value_analog(value)
            MyCtr.left_wheel = MyCtr.right_wheel= 1
            data = [MyCtr.left_pwm, MyCtr.left_wheel, MyCtr.right_pwm, MyCtr.right_wheel]
            bus.write_i2c_block_data(address, 0, data)

    def on_L3_right(self, value):
        MyCtr.turn_left = False
        MyCtr.turn_right = True
        #print("drive_forward: ", MyCtr.drive_forward, ", drive_reverse: ", MyCtr.drive_reverse)
        if not (MyCtr.drive_forward == MyCtr.drive_reverse):
            MyCtr.set_right_pwm(value)
        else:
            MyCtr.left_pwm = MyCtr.right_pwm = MyCtr.get_motor_value_analog(value)
            MyCtr.left_wheel = MyCtr.right_wheel = 0
            data = [MyCtr.left_pwm, MyCtr.left_wheel, MyCtr.right_pwm, MyCtr.right_wheel]
            bus.write_i2c_block_data(address, 0, data)
    
    def on_L3_up(self, value):
        pass 

    def on_L3_down(self, value):
        pass 

    def on_L3_y_at_rest(self):
        MyCtr.turn_left = False
        MyCtr.turn_right = False
        if MyCtr.drive_forward == MyCtr.drive_reverse:
            MyCtr.left_pwm = MyCtr.right_pwm = 0
            data = [MyCtr.left_pwm, MyCtr.left_wheel, MyCtr.right_pwm, MyCtr.right_wheel]
            bus.write_i2c_block_data(address, 0, data)
    
    def on_L3_x_at_rest(self):
        MyCtr.turn_left = False
        MyCtr.turn_right = False
        if MyCtr.drive_forward == MyCtr.drive_reverse:
            MyCtr.left_pwm = MyCtr.right_pwm = 0
            data = [MyCtr.left_pwm, MyCtr.left_wheel, MyCtr.right_pwm, MyCtr.right_wheel]
            bus.write_i2c_block_data(address, 0, data)
        

    # Do nothing commands
    # def on_x_release(self):
    #     pass    

    # #def on_L3_up(self, value):
    #     pass

    # #def on_L3_down(self, value):
    #     pass

    # def on_L3_y_at_rest(self):
    #     pass

    # def on_L3_x_at_rest(self):
    #     pass

    # def on_R3_right(self, value):
    #     pass

    # def on_R3_left(self, value):
    #     pass

    # def on_R3_y_at_rest(self):
    #     pass

    # def on_R3_x_at_rest(self):
    #     pass

atexit.register(goodbye)

# If gpio 16 is postive and gpio 26 is negative, the engine turns one way. If changes it drives the other way - make a specific button for this. PWM signal value is Gpio 13

# Gpio 12 is pwm for turning. Gpio 5 and 6 is used to determine which direction.

controller = MyCtr(interface="/dev/input/js0", connecting_using_ds4drv=False)
# you can start listening before controller is paired, as long as you pair it within the timeout window

controller.listen(timeout=60)   

        
"""
    def buzz(noteFreq, duration):
        halveWaveTime = 1 / (noteFreq * 2 )
        waves = int(duration * noteFreq)
        for i in range(waves):
            GPIO.output(MyController.horn, True)
            time.sleep(halveWaveTime)
            GPIO.output(MyController.horn, False)
            time.sleep(halveWaveTime)

    def on_triangle_press(self):
        t=0
        notes=[262,294,330,262,262,294,330,262,330,349,392,330,349,392,392,440,392,349,330,262,392,440,392,349,330,262,262,196,262,262,196,262]
        duration=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,1,0.5,0.5,1,0.25,0.25,0.25,0.25,0.5,0.5,0.25,0.25,0.25,0.25,0.5,0.5,0.5,0.5,1,0.5,0.5,1]
        for n in notes:
            MyController.buzz(n, duration[t])
            time.sleep(duration[t] *0.1)
            t+=1
"""