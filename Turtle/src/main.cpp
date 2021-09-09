#include <Arduino.h>
#include <Wire.h>
#include "Modules/Motor.hpp"

// Pinout
// Bumper
const uint8_t bumper_left = 0;
const uint8_t bumper_center = 1;
const uint8_t bumper_right = 2;

// Horn
const uint8_t horn = 10;

// Motor instances
const motor::Motor wheel_left(3, 12);
const motor::Motor wheel_right(11, 13);

bool is_bumper_pressed()
{
  bool is_left = digitalRead(bumper_left);
  bool is_center = digitalRead(bumper_center);
  bool is_right = digitalRead(bumper_right);

  bool error_bip = (!is_left || !is_center || !is_right) ? true : false;

  if (!is_left || !is_center || !is_right)
  {
    wheel_left.runPWM(0, motor::direction::Forward);
    wheel_right.runPWM(0, motor::direction::Forward);

    // Indicate an error
    digitalWrite(horn, HIGH);
    return true;
  }
  else
  {
    digitalWrite(horn, LOW);
    return false;
  }
}

void receiveEvent(int howMany)
{
  // Dummy read
  Wire.read();

  // Read the 4 incoming bytes
  // order: [left_pwm, left_dir, left_dir, left_pwm]
  if ((Wire.available() == 4) && (!is_bumper_pressed()))
  {
    wheel_left.runPWM(Wire.read(), static_cast<motor::direction>(Wire.read()));
    wheel_right.runPWM(Wire.read(), static_cast<motor::direction>(Wire.read()));
  }
  else
  {
    // Clear buffer
    while (Wire.available())
    {
      // Read dummy
      Wire.read();
    }
  }
}

void requestEvent()
{
  // Return if bumper is pressed
  Wire.write(is_bumper_pressed());
}

void setup()
{
  // Begin serial
  Serial.begin(115200);

  // Set bumper values
  pinMode(bumper_left, INPUT);
  pinMode(bumper_center, INPUT);
  pinMode(bumper_right, INPUT);
  pinMode(horn, OUTPUT);
  digitalWrite(horn, LOW);

  // Configurate wheel PWM
  analogWriteFrequency(31250);
  analogWriteResolution(8);

  // Initialize wheels
  wheel_left.runPWM(0, motor::direction::Forward);
  wheel_right.runPWM(0, motor::direction::Forward);

  // Configurate I2C
  Wire.begin(25);
  Wire.onRequest(requestEvent);
  Wire.onReceive(receiveEvent);
}

const uint16_t delay_msec = 500;
const uint8_t left = 0;
const uint8_t right = 0;
uint8_t counter = 25;
void loop()
{
  is_bumper_pressed();
  delay(10);
}
