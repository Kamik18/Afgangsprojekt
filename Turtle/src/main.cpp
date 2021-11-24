#include "Modules/Encoder/Encoder.hpp"
#include "Modules/Motor/Motor.hpp"
#include "Modules/Sound/Sound.hpp"
#include "Modules/PID/PID.hpp"

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
const uint8_t enc_right = 7;

// Motor instances
const motor::Motor wheel_left(3, 12);
const motor::Motor wheel_right(11, 13);
uint32_t           counter = 0;

void pid::regulator() {
    left_pid.set_input((encoder::distance_left * 100) / encoder::time_span);
    right_pid.set_input((encoder::distance_right * 100) / encoder::time_span);

    left_pid.pid.Compute();
    right_pid.pid.Compute();

    wheel_left.runPWM(left_pid.get_output(), motor::direction::Forward);
    wheel_right.runPWM(right_pid.get_output(), motor::direction::Reverse);

    // Serial.println(String(counter) + "," + String(left_pid.get_setpoint()) + "," + String(left_pid.get_input()) + ","
    // +
    //                String(right_pid.get_input()));
    counter++;
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
        pid::left_pid.set_output(0);
        pid::right_pid.set_output(0);
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
void set_speed(const uint8_t lin, const uint8_t ang_p, const uint8_t ang_n) {
    const double lin_vel = lin / 255.0;

    const double limit   = 60.0;
    const double ang_vel = (static_cast<double>(ang_p - ang_n) / 255.0) * (encoder::distance_between_wheel / 2.0);
    double       left    = constrain(((lin_vel - ang_vel) * limit), 0.0, limit);
    double       right   = constrain(((lin_vel + ang_vel) * limit), 0.0, limit);

    pid::set_setpoint(&pid::left_pid, left);
    pid::set_setpoint(&pid::right_pid, right);
}

void receiveEvent(int howMany) {
    // Dummy read
    Wire.read();

    if ((Wire.available() == 3) && (!is_bumper_pressed())) {
        // Update speed
        set_speed(Wire.read(), Wire.read(), Wire.read());
    } else if (Wire.available() == 1) {
        // Set flag
        sound::is_play_sound = true;
    }

    // Clear buffer
    while (Wire.available()) {
        // Read dummy
        Wire.read();
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

    const double lower = 20;
    const double upper = 255;
    pid::left_pid.pid.SetOutputLimits(lower, upper);
    pid::right_pid.pid.SetOutputLimits(lower, upper);

    pid::left_pid.pid.SetSampleTime(20);
    pid::right_pid.pid.SetSampleTime(20);

    // Start the hardwaretimer
    encoder::timer.attachInterrupt(encoder::velocity);
    encoder::timer.setOverflow(20, HERTZ_FORMAT);
    encoder::timer.resume();

    pid::timer.attachInterrupt(pid::regulator);
    pid::timer.setOverflow(20, HERTZ_FORMAT);
    pid::timer.resume();
}

void loop() {
    is_bumper_pressed();
    delay(10);

    // Check flag
    if (sound::is_play_sound) {
        // Play a sound
        sound::play_sound(horn);
    }

    /*
    if (counter < 500) {
        pid::set_setpoint(&pid::left_pid, 60);
        pid::set_setpoint(&pid::right_pid, 60);
        delay(5000);
        pid::set_setpoint(&pid::left_pid, 0);
        pid::set_setpoint(&pid::right_pid, 0);
        delay(5000);
    } else {
        pid::set_setpoint(&pid::left_pid, 0);
        pid::set_setpoint(&pid::right_pid, 0);
        delay(2000);
    }
    */
}
