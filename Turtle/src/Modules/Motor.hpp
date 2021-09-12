#include <Arduino.h>

namespace motor {
    enum class direction : uint8_t
    {
        Forward = 0,
        Reverse = 1,
    };

    class Motor {
      public:
        Motor(const uint8_t pin_pwm, const uint8_t pin_dir) : _pin_pwm(pin_pwm), _pin_dir(pin_dir) {
            pinMode(_pin_pwm, OUTPUT);
            pinMode(_pin_dir, OUTPUT);
        }

        void runPWM(const uint32_t pwm, const direction dir) const {
            analogWrite(_pin_pwm, pwm);
            digitalWrite(_pin_dir, static_cast<uint8_t>(dir));
        }

      private:
        const uint8_t _pin_pwm;
        const uint8_t _pin_dir;
    };
} // namespace motor