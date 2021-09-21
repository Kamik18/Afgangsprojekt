#include <Arduino.h>
#include <PID_v1.h>

namespace pid {
    // Create a hardwaretimer
    HardwareTimer timer(TIM16);

    // PID struct
    double Kp = 2, Ki = 1, Kd = 0.25;
    struct PidStruct {
        double input;
        double output;
        double setpoint;

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
