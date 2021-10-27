#include "PID/PID_v1.h"
#include <Arduino.h>

namespace pid {
    // Create a hardwaretimer
    HardwareTimer timer(TIM16);
    class PID_controller {
      public:
        PID_controller(const double Kp, const double Ki, const double Kd) {
            pid = PID(&this->_input, &this->_output, &this->_setpoint, Kp, Ki, Kd, DIRECT);
        }

        PID pid;

        void   set_input(const double input) { this->_input = input; }
        double get_input() const { return this->_input; }

        void   set_output(const double output) { this->_output = output; }
        double get_output() const { return this->_output; }

        void   set_setpoint(const double setpoint) { this->_setpoint = setpoint; }
        double get_setpoint() const { return this->_setpoint; }

      private:
        double _input    = 0;
        double _output   = 0;
        double _setpoint = 0;
    };

    // PID
    const double   Kp = 3;
    const double   Ki = 2;
    const double   Kd = 0.05;
    PID_controller left_pid(Kp + 0.5, Ki, Kd);
    PID_controller right_pid(Kp, Ki, Kd);

    // Kp 1 = 300 s
    // Kp 2 = xx s
    // Kp 3 = 200 s
    // Kp 4 = 187 s

    extern void regulator();

    void set_setpoint(PID_controller *pid_controller, const double setpoint) { pid_controller->set_setpoint(setpoint); }
} // namespace pid
