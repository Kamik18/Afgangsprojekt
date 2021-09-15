#include <Arduino.h>

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
    const double   wheel_radius = 0.145 / 2;
    const double   wheel_circ   = wheel_radius * 2 * PI;
    // Conversion to 1 tick = 0.0003 m
    const double tick_conversion = (wheel_circ / static_cast<double>(ecoder_ticks));

    // Measured in meter
    const double distance_between_wheel = 0.29;

    // Position in meter
    double state_x      = 0;
    double state_y      = 0;
    double state_theata = 0;

    void velocity() {
        uint32_t ticks     = HAL_GetTick();
        double   time_span = static_cast<double>(ticks - prev_ticks) / 1000;
        Serial.println("Ticks: " + String(enc_left) + ", " + String(enc_right));

        // Measured in meters
        double distance_left  = tick_conversion * static_cast<double>(enc_left);
        double distance_right = tick_conversion * static_cast<double>(enc_right);

        // Clear encoders
        enc_left  = 0;
        enc_right = 0;

        // Delta theta
        const double delta_theta       = (distance_right - distance_left) / (distance_between_wheel);
        const double rotation_velocity = delta_theta / time_span;
        Serial.println("Rotation: " + String(rotation_velocity) + " rad/s");

        // Distance traveled
        const double dist_traveled = (distance_left - distance_right) / 2;
        Serial.println("Distance: " + String(dist_traveled * 100) + " cm");

        // Update states
        state_x      = state_x + (dist_traveled * cos(state_theata));
        state_y      = state_y + (dist_traveled * sin(state_theata));
        state_theata = state_theata + delta_theta;
        Serial.println("x: " + String(state_x) + " m");
        Serial.println("y: " + String(state_y) + " m");
        Serial.println("theata: " + String(state_theata) + " rad");
        Serial.println();
    }
} // namespace encoder
