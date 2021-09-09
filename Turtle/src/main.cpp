#include <Arduino.h>
//#include <EEPROM.h>

//#include <MotorWheel.h>
//#include <R2WD.h>

//#include <fuzzy_table.h>
//#include <PID_Beta6.h>

//#include <SONAR.h>
#include "Modules/Motor.hpp"

// Sensor positions
//         - - - - - - - - - - - - - - - -
//        /       power switch            \
//       /                                 \
//      |                                   |
//      |                                   |
//      |  wheel2                    wheel1 |
//      |                                   |
//      |                                   |
//       \                                  /
//        \  sonar3     sonar 2    sonar 1 /
//         \                              /
//          \                            /
//            -  -  -  -  -  -  -  -  -
//          bumper_R    bumper_C    bumper_L

/******************************************/
/*
// SONAR
SONAR sonar11(0x11),sonar12(0x12),sonar13(0x13);

unsigned short distBuf[3];
void sonarsUpdate() {
    static unsigned char sonarCurr=1;
    if(sonarCurr==3) sonarCurr=1;
    else ++sonarCurr;
    if(sonarCurr==1) {        
        distBuf[1]=sonar12.getDist();        
        sonar12.trigger();        
    } else if(sonarCurr==2) {
        distBuf[2]=sonar13.getDist();
        sonar13.trigger();
    } else {
        distBuf[0]=sonar11.getDist();
        sonar11.trigger();
    }
}
*/

/*********************************************/
// Optional device
// Infrared Sensor
/*
const uint8_t irL0_pin=0;    // Analog
const uint8_t irC0_pin=1;
const uint8_t irR0_pin=2;

int ir_distance(unsigned char ir) {
	int val=analogRead(ir);
	return (6762/(val-9))-4;
}
*/

// Bumper
uint8_t bumper_left = 1;
uint8_t bumper_center = 2;
uint8_t bumper_right = 3;

motor::IRQ_struct irq_left;
motor::IRQ_struct irq_right;
// motor::Motor wheel_left(9, 8, /*4, 5,*/ &irq_left);
// motor::Motor wheel_right(10, 11, /*6, 7,*/ &irq_right);
motor::Motor wheel_left(3, 12, /*4, 5,*/ &irq_left);
motor::Motor wheel_right(11, 13, /*6, 7,*/ &irq_right);

uint8_t horn = 0;

void stop_wheels()
{
  wheel_left.runPWM(0, motor::direction::Forward);
  wheel_right.runPWM(0, motor::direction::Forward);
  delay(250);
}

void is_bumper_pressed()
{
  bool is_left = digitalRead(bumper_left);
  bool is_center = digitalRead(bumper_center);
  bool is_right = digitalRead(bumper_right);

  bool error_bip = (!is_left || !is_center || !is_right) ? true : false;

  digitalWrite(horn, error_bip);

  Serial.println("Left: " + String(is_left) + "\t\t" +
                 "Center: " + String(is_center) + "\t" +
                 "Right: " + String(is_right) + "\t\t" +
                 "Beep: " + String(error_bip));
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
}

const uint16_t delay_msec = 500;
const uint8_t left = 0;
const uint8_t right = 0;
uint8_t counter = 0;
void loop()
{
  wheel_left.runPWM(left + counter, motor::direction::Forward);
  wheel_right.runPWM(right + counter, motor::direction::Forward);
  delay(delay_msec);
  stop_wheels();
  delay(delay_msec);
  Serial.println(counter);
  counter++;

  is_bumper_pressed();
  delay(100);
}
