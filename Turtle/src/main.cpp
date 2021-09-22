
#include "Modules/Encoder.hpp"
#include "Modules/Motor.hpp"
#include "Modules/PID.hpp"
#include <Arduino.h>
#include <HardwareTimer.h>
#include <Wire.h>

// Pinout
// Bumper
const uint8_t bumper_left   = 0;
const uint8_t bumper_center = 1;
const uint8_t bumper_right  = 2;

// Horn
const uint8_t horn = 10;

// Encoder
const uint8_t enc_left  = 4;
const uint8_t enc_right = 5;

// Motor instances
const motor::Motor wheel_left(3, 12);
const motor::Motor wheel_right(11, 13);

pid::PidStruct *test_pid = &pid::left_pid;

void pid::regulator() {
    left_pid.input  = encoder::distance_left / encoder::time_span;
    right_pid.input = encoder::distance_right / encoder::time_span;

    left_pid.pid.Compute();
    right_pid.pid.Compute();

    wheel_left.runPWM(left_pid.output * 100, motor::direction::Forward);
    wheel_right.runPWM(right_pid.output * 100, motor::direction::Reverse);

    const uint16_t factor = 1000;
    // Serial.println(String(test_pid->setpoint * factor) + "," + String(test_pid->input * factor));
}

bool is_bumper_pressed() {
    bool is_left   = digitalRead(bumper_left);
    bool is_center = digitalRead(bumper_center);
    bool is_right  = digitalRead(bumper_right);

    bool error_bip = (!is_left || !is_center || !is_right) ? true : false;

    if (!is_left || !is_center || !is_right) {
        wheel_left.runPWM(0, motor::direction::Forward);
        wheel_right.runPWM(0, motor::direction::Forward);

        // Indicate an error
        digitalWrite(horn, HIGH);

        // Update PID setpoint
        pid::set_setpoint(&pid::left_pid, 0);
        pid::set_setpoint(&pid::right_pid, 0);
        pid::left_pid.output  = 0;
        pid::right_pid.output = 0;
        return true;
    } else {
        digitalWrite(horn, LOW);
        return false;
    }
}

//-----------------------------------------------------------
// @brief Set the speed setpoint for both motors.
// @param lin The linear velocity.
// @param ang The angular velocity.
void set_speed(double lin, double ang) {
    wheel_left.runPWM(Wire.read(), motor::direction::Forward);
    wheel_right.runPWM(Wire.read(), motor::direction::Reverse);

    lin /= 100;
    ang /= 1000;
    const double ang_vel = ang * (encoder::distance_between_wheel / 2);

    double left  = lin - ang_vel;
    double right = lin + ang_vel;

    if (left < 0) {
        left = 0;
    }
    if (right < 0) {
        right = 0;
    }
    if (left > 0.8) {
        left = 0.8;
    }
    if (right > 0.8) {
        right = 0.8;
    }
    pid::set_setpoint(&pid::left_pid, left);
    pid::set_setpoint(&pid::right_pid, right);

    Serial.println("lin: " + String(lin) + ", ang_vel: " + String(ang_vel) + ", left: " + String(left) +
                   ", right: " + String(right));
}

void receiveEvent(int howMany) {
    // Dummy read
    Wire.read();

    // Read the 4 incoming bytes
    // order: [left_pwm, left_dir, right_dir, right_pwm]
    if ((Wire.available() == 2) && (!is_bumper_pressed())) {
        set_speed(Wire.read(), Wire.read());
    } else if ((Wire.available() == 4) && (!is_bumper_pressed())) {
        wheel_left.runPWM(Wire.read(), static_cast<motor::direction>(Wire.read()));
        wheel_right.runPWM(Wire.read(), static_cast<motor::direction>(Wire.read()));
    } else {
        // Clear buffer
        while (Wire.available()) {
            // Read dummy
            Wire.read();
        }
    }
}

void requestEvent() {
    encoder::state_x;
    encoder::state_y;
    encoder::state_theata;
    std::array<uint8_t, 1 + (3 * sizeof(double))> data;

    data.at(0) = is_bumper_pressed();
    memcpy(&data.at(1 + 0 * sizeof(double)), &encoder::state_x, sizeof(double));
    memcpy(&data.at(1 + 1 * sizeof(double)), &encoder::state_y, sizeof(double));
    memcpy(&data.at(1 + 2 * sizeof(double)), &encoder::state_theata, sizeof(double));

    // Return if bumper is pressed
    Wire.write(data.data(), data.size());
}

void setup() {
    // Begin serial
    Serial.begin(115200);

    // Set pinouts
    // Bumper
    pinMode(bumper_left, INPUT);
    pinMode(bumper_center, INPUT);
    pinMode(bumper_right, INPUT);
    // Horn
    pinMode(horn, OUTPUT);
    digitalWrite(horn, LOW);
    // Encoder
    pinMode(enc_left, INPUT_PULLUP);
    pinMode(enc_right, INPUT_PULLUP);

    // Attach interrupts to the digital pins
    attachInterrupt(enc_left, encoder::isr_left, CHANGE);
    attachInterrupt(enc_right, encoder::isr_right, CHANGE);

    // Configurate wheel PWM
    analogWriteFrequency(31250);
    analogWriteResolution(8);

    // Initialize wheels
    wheel_left.runPWM(0, motor::direction::Forward);
    wheel_right.runPWM(0, motor::direction::Forward);

    // Configurate I2C
    Wire.begin(25);
    Wire.onRequest(requestEvent);
    Wire.onReceive(receiveEvent);

    // Turn the PID on
    pid::left_pid.pid.SetMode(AUTOMATIC);
    pid::right_pid.pid.SetMode(AUTOMATIC);
    pid::left_pid.pid.SetOutputLimits(0, 2.55);
    pid::right_pid.pid.SetOutputLimits(0, 2.55);
    set_speed((0.0 * 100), 0);
    pid::set_setpoint(test_pid, 0);

    // Start the hardwaretimer
    encoder::timer.attachInterrupt(encoder::velocity);
    encoder::timer.setOverflow(10, HERTZ_FORMAT);
    encoder::timer.resume();

    pid::timer.attachInterrupt(pid::regulator);
    pid::timer.setOverflow(10, HERTZ_FORMAT);
    pid::timer.resume();
}

void loop() {
    // is_bumper_pressed();
    // delay(100);

    pid::set_setpoint(test_pid, 0.4);
    delay(10000);
    pid::set_setpoint(test_pid, 0.2);
    delay(10000);
    pid::set_setpoint(test_pid, 0);
    delay(10000);

    // if (encoder::state_x > 4) {
    //    pid::set_setpoint(&pid::left_pid, 0);
    //    pid::set_setpoint(&pid::right_pid, 0);
    //}
}
