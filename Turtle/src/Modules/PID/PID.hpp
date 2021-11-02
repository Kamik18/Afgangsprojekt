//===================================================================
// File: PID.hpp
//===================================================================
#pragma once

//-------------------------------------------------------------------
// Includes
//-------------------------------------------------------------------
#include <Arduino.h>
#include "PID_v1.h"

//-------------------------------------------------------------------
// PID namespace
//-------------------------------------------------------------------
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
    const double   Kp = 0.5;
    const double   Ki = 1.5;
    const double   Kd = 0.01;
    PID_controller left_pid(Kp * 1.13, Ki * 1.13, Kd * 1.13);
    PID_controller right_pid(Kp, Ki, Kd);

    extern void regulator();

    void set_setpoint(PID_controller *pid_controller, const double setpoint) { pid_controller->set_setpoint(setpoint); }
} // namespace pid
