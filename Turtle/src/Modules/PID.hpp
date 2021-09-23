#include <Arduino.h>
#include "PID/PID_v1.h"

namespace pid {
    // Create a hardwaretimer
    HardwareTimer timer(TIM16);

    // PID struct
    //double Kp = 1.5, Ki = 2, Kd = 0.05;
    double Kp = 3, Ki = 2, Kd = 0.05;
    struct PidStruct {
        double input;
        double output;
        double setpoint = 0;

        PID pid = PID(&input, &output, &setpoint, Kp, Ki, Kd, DIRECT);
    };


    // PID 
    PidStruct left_pid;
    PidStruct right_pid;

    extern void regulator();

    void set_setpoint(PidStruct *pid_struct, const double value) {
        pid_struct->setpoint = value;
    }
} // namespace pid
