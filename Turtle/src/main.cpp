
#include "Modules/Encoder.hpp"
#include "Modules/Motor.hpp"
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
    if ((Wire.available() == 4) && (!is_bumper_pressed())) {
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
    // Return if bumper is pressed
    Wire.write(is_bumper_pressed());
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

    // Start the hardwaretimer
    encoder::timer.attachInterrupt(encoder::velocity);
    encoder::timer.setOverflow(2, HERTZ_FORMAT);
    encoder::timer.resume();
}

const uint16_t delay_msec = 500;
const uint8_t  left       = 0;
const uint8_t  right      = 0;
uint8_t        counter    = 0;
void           loop() {
    is_bumper_pressed();
    delay(10);
}
