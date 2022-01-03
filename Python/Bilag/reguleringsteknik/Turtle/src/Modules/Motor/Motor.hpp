//===================================================================
// File: Motor.hpp
//===================================================================
#pragma once

//-------------------------------------------------------------------
// Includes
//-------------------------------------------------------------------
#include <Arduino.h>

//-------------------------------------------------------------------
// Motor namespace
//-------------------------------------------------------------------
namespace motor {
    enum class direction : uint8_t
    {
        Forward = 0,
        Reverse = 1,
    };

    class Motor {
      public:
        Motor(const uint8_t pin_pwm, const uint8_t pin_dir);

        void runPWM(const uint32_t pwm, const direction dir) const;

      private:
        const uint8_t _pin_pwm;
        const uint8_t _pin_dir;
    };
} // namespace motor