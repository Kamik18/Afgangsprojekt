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
    // Wheel circumference = 14.5 cm * pi = 45.55309 cm
    // 1536 ticks = 45.55309 cm
    const uint16_t ecoder_ticks = 1536;
    const double   wheel_circ   = 45.55309;
    // Conversion to 1 tick = 0.03 cm
    const double tick_conversion = (wheel_circ / static_cast<double>(1536));

    void velocity() {
        uint32_t ticks     = HAL_GetTick();
        double   time_span = static_cast<double>(ticks - prev_ticks) / 1000;
        Serial.println("Ticks: " + String(enc_left) + ", " + String(enc_right));

        // Measured in centimeters
        double distance_left  = tick_conversion * static_cast<double>(enc_left);
        double distance_right = tick_conversion * static_cast<double>(enc_right);
        // Measured in cm/s
        double velocity_left  = distance_left / time_span;
        double velocity_right = distance_right / time_span;

        // Clear encoders
        enc_left  = 0;
        enc_right = 0;

        // Update previously values
        prev_ticks = ticks;

        // Temp
        Serial.println("Distance: " + String(distance_left) + ": " + String(distance_right));
        Serial.println("Velocity: " + String(velocity_left) + ": " + String(velocity_right));
        Serial.println();
    }
} // namespace encoder