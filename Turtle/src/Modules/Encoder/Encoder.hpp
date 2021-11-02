//===================================================================
// File: Encoder.hpp
//===================================================================
#pragma once

//-------------------------------------------------------------------
// Includes
//-------------------------------------------------------------------
#include <Arduino.h>

//-------------------------------------------------------------------
// Encoder namespace
//-------------------------------------------------------------------
namespace encoder {
    // Create a hardwaretimer
    extern HardwareTimer timer;

    // Interrupt functions
    extern void isr_left();
    extern void isr_right();

    // Measured in meter
    extern const double distance_between_wheel;

    // Position in meter
    extern double state_x;
    extern double state_y;
    // double state_theata   = -0.88761753;
    extern double state_theata  ;
    extern double time_span     ;
    extern double distance_left ;
    extern double distance_right;

    extern void velocity();
} // namespace encoder
