
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

void pid::regulator() {
    left_pid.input  = encoder::distance_left / encoder::time_span;
    right_pid.input = encoder::distance_left / encoder::time_span;
    left_pid.pid.Compute();
    right_pid.pid.Compute();
    wheel_left.runPWM(left_pid.output * 100, motor::direction::Forward);
    wheel_right.runPWM(right_pid.output * 100, motor::direction::Reverse);
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
        pid::left_pid.output = 0;
        pid::set_setpoint(&pid::right_pid, 0);
        pid::right_pid.output = 0;
        return true;
    } else {
        digitalWrite(horn, LOW);
        return false;
    }
}

void receiveEvent(int howMany) {
    // Dummy read
    Wire.read();

    // Read the 4 incoming bytes
    // order: [left_pwm, left_dir, left_dir, left_pwm]
    if ((Wire.available() == 2) && (!is_bumper_pressed())) {
        wheel_left.runPWM(Wire.read(), motor::direction::Forward);
        wheel_right.runPWM(Wire.read(), motor::direction::Reverse);
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
    pid::set_setpoint(&pid::left_pid, 0.4);
    pid::set_setpoint(&pid::right_pid, 0.4);

    // Start the hardwaretimer
    encoder::timer.attachInterrupt(encoder::velocity);
    encoder::timer.setOverflow(10, HERTZ_FORMAT);
    encoder::timer.resume();

    pid::timer.attachInterrupt(pid::regulator);
    pid::timer.setOverflow(10, HERTZ_FORMAT);
    pid::timer.resume();
}

void loop() {
    is_bumper_pressed();
    delay(10);

    if (encoder::state_x > 4) {
        pid::set_setpoint(&pid::left_pid, 0);
        pid::set_setpoint(&pid::right_pid, 0);
    }
}
