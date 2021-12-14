//===================================================================
// File: Encoder.cpp
//===================================================================
//-------------------------------------------------------------------
// Includes
//-------------------------------------------------------------------
#include "Encoder.hpp"

namespace encoder {
    // Create a hardwaretimer
    HardwareTimer timer(TIM2);

    size_t enc_left  = 0;
    size_t enc_right = 0;
    void   isr_left() { ++enc_left; }
    void   isr_right() { ++enc_right; }

    uint32_t prev_ticks = HAL_GetTick();

    // Gear ratio = 64
    // Encoder ticks = 12 -> read on changes = 24 ticks pr revolution
    // 1 wheel revolution = 64 * 24 = 1536 ticks
    // Wheel circumference = 0.145 m * pi = 0.46 m
    // 1536 ticks = 0.46 m
    const uint16_t ecoder_ticks = 64 * 24;
    const double   wheel_radius = 0.143 / 2;
    const double   wheel_circ   = wheel_radius * 2 * PI * 1.022;
    // Conversion to 1 tick = 0.0003 m
    const double tick_conversion = (wheel_circ / static_cast<double>(ecoder_ticks));

    // Measured in meter
    const double distance_between_wheel = 0.29;

    // Position in meter
    double state_x = 0;
    double state_y = 0;
    // double state_theata   = -0.88761753;
    double state_theata   = 0;
    double time_span      = 0;
    double distance_left  = 0;
    double distance_right = 0;

    void velocity() {
        uint32_t ticks = HAL_GetTick();
        time_span      = static_cast<double>(ticks - prev_ticks) / 1000;
        prev_ticks     = ticks;

        // Measured in meters
        distance_left  = tick_conversion * static_cast<double>(enc_left);
        distance_right = tick_conversion * static_cast<double>(enc_right);

        // Clear encoders
        enc_left  = 0;
        enc_right = 0;

        // Delta theta
        const double delta_theta = (distance_right - distance_left) / (distance_between_wheel);

        // Distance traveled
        const double dist_traveled = (distance_left + distance_right) / 2;

        // Update states
        state_x      = state_x + (dist_traveled * cos(state_theata));
        state_y      = state_y + (dist_traveled * sin(state_theata));
        state_theata = state_theata + delta_theta;

        if (state_theata < (-2.0 * PI)) {
            state_theata += (2.0 * PI);
        } else if (state_theata > (2.0 * PI)) {
            state_theata -= (2.0 * PI);
        }
    }
} // namespace encoder
