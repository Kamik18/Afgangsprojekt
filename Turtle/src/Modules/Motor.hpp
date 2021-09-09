#include <Arduino.h>

namespace motor
{
    enum class direction : uint8_t
    {
        Reverse = 0,
        Forward = 1
    };

    struct IRQ_struct
    {
        void (*irq_func)();
        uint8_t _pin_irq_a;
        uint8_t _pin_irq_b;
    };

    class Motor
    {
    public:
        Motor(const uint8_t pin_pwm, const uint8_t pin_dir,
              //const uint8_t pin_irq_a, const uint8_t pin_irq_b,
              struct IRQ_struct *irq)
            : _pin_pwm(pin_pwm), _pin_dir(pin_dir), _irq(irq)
        {

            //_irq->_pin_irq_a = pin_irq_a;
            //_irq->_pin_irq_b = pin_irq_b;

            pinMode(_pin_pwm, OUTPUT);
            pinMode(_pin_dir, OUTPUT);
            pinMode(_irq->_pin_irq_a, INPUT);
            pinMode(_irq->_pin_irq_b, INPUT);
        }

        void runPWM(const uint32_t pwm, const direction dir)
        {
            analogWrite(_pin_pwm, pwm);
            digitalWrite(_pin_dir, static_cast<uint8_t>(dir));
        }

        struct IRQ_struct *_irq;

    private:
        uint8_t _pin_pwm;
        uint8_t _pin_dir;
    };
}