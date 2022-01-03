//===================================================================
// File: Motor.cpp
//===================================================================
//-------------------------------------------------------------------
// Includes
//-------------------------------------------------------------------
#include "Motor.hpp"

namespace motor {
    Motor::Motor(const uint8_t pin_pwm, const uint8_t pin_dir) : _pin_pwm(pin_pwm), _pin_dir(pin_dir) {
        pinMode(this->_pin_pwm, OUTPUT);
        pinMode(this->_pin_dir, OUTPUT);
    }

    void Motor::runPWM(const uint32_t pwm, const direction dir) const {
        analogWrite(this->_pin_pwm, pwm);
        digitalWrite(this->_pin_dir, static_cast<uint8_t>(dir));
    }
} // namespace motor
