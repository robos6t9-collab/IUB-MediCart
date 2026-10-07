#include <Arduino.h>

// ==========================================
// MOTOR PIN DEFINITIONS
// ==========================================

// Left Motor - BTS Driver
const int MOTOR_L_PIN1 = 2;   // Forward PWM
const int MOTOR_L_PIN2 = 3;   // Reverse PWM

// Right Motor - BTS Driver
const int MOTOR_R_PIN1 = 4;   // Forward PWM
const int MOTOR_R_PIN2 = 5;   // Reverse PWM


// ==========================================
// 5-CHANNEL IR SENSOR ARRAY
// ==========================================

// [0] Far Left
// [1] Left
// [2] Center
// [3] Right
// [4] Far Right

const int IR_PINS[5] = {A8, A9, A10, A11, A12};


// ==========================================
// SETTINGS
// ==========================================

// true  = black line on white surface
// false = white line on black surface
const bool BLACK_LINE = true;


// Motor speeds
const int FORWARD_SPEED = 150;
const int SLIGHT_SPEED  = 120;
const int STRONG_SPEED = 180;


// ==========================================
// MOTOR CONTROL
// ==========================================

void leftMotor(int speed)
{
    if (speed > 0)
    {
        analogWrite(MOTOR_L_PIN1, speed);
        analogWrite(MOTOR_L_PIN2, 0);
    }
    else if (speed < 0)
    {
        analogWrite(MOTOR_L_PIN1, 0);
        analogWrite(MOTOR_L_PIN2, -speed);
    }
    else
    {
        analogWrite(MOTOR_L_PIN1, 0);
        analogWrite(MOTOR_L_PIN2, 0);
    }
}


void rightMotor(int speed)
{
    if (speed > 0)
    {
        analogWrite(MOTOR_R_PIN1, speed);
        analogWrite(MOTOR_R_PIN2, 0);
    }
    else if (speed < 0)
    {
        analogWrite(MOTOR_R_PIN1, 0);
        analogWrite(MOTOR_R_PIN2, -speed);
    }
    else
    {
        analogWrite(MOTOR_R_PIN1, 0);
        analogWrite(MOTOR_R_PIN2, 0);
    }
}


void forward()
{
    leftMotor(FORWARD_SPEED);
    rightMotor(FORWARD_SPEED);
}


void slightRight()
{
    leftMotor(SLIGHT_SPEED);
    rightMotor(FORWARD_SPEED);
}


void strongRight()
{
    leftMotor(STRONG_SPEED);
    rightMotor(0);
}


void slightLeft()
{
    leftMotor(FORWARD_SPEED);
    rightMotor(SLIGHT_SPEED);
}


void strongLeft()
{
    leftMotor(0);
    rightMotor(STRONG_SPEED);
}


void stopMotors()
{
    leftMotor(0);
    rightMotor(0);
}


// ==========================================
// READ SENSOR
// ==========================================

bool sensorActive(int index)
{
    int value = digitalRead(IR_PINS[index]);

    // Reverse sensor logic depending on line color
    if (BLACK_LINE)
    {
        return value == LOW;
    }
    else
    {
        return value == HIGH;
    }
}


// ==========================================
// SETUP
// ==========================================

void setup()
{
    Serial.begin(115200);

    for (int i = 0; i < 5; i++)
    {
        pinMode(IR_PINS[i], INPUT);
    }

    pinMode(MOTOR_L_PIN1, OUTPUT);
    pinMode(MOTOR_L_PIN2, OUTPUT);
    pinMode(MOTOR_R_PIN1, OUTPUT);
    pinMode(MOTOR_R_PIN2, OUTPUT);

    stopMotors();

    Serial.println("5-Sensor Line Follower Started");
}


// ==========================================
// MAIN LOOP
// ==========================================

void loop()
{
    bool s0 = sensorActive(0);
    bool s1 = sensorActive(1);
    bool s2 = sensorActive(2);
    bool s3 = sensorActive(3);
    bool s4 = sensorActive(4);


    // ======================================
    // PRINT SENSOR PATTERN
    // ======================================

    Serial.print(s0);
    Serial.print(s1);
    Serial.print(s2);
    Serial.print(s3);
    Serial.println(s4);


    // ======================================
    // LINE FOLLOWING LOGIC
    // ======================================

    // 00100
    // Centered -> FORWARD
    if (!s0 && !s1 && s2 && !s3 && !s4)
    {
        forward();
    }

    // 01100
    // Slight RIGHT
    else if (!s0 && s1 && s2 && !s3 && !s4)
    {
        slightRight();
    }

    // 11000
    // Strong RIGHT
    else if (s0 && s1 && !s2 && !s3 && !s4)
    {
        strongRight();
    }

    // 00110
    // Slight LEFT
    else if (!s0 && !s1 && s2 && s3 && !s4)
    {
        slightLeft();
    }

    // 00011
    // Strong LEFT
    else if (!s0 && !s1 && !s2 && s3 && s4)
    {
        strongLeft();
    }

    // ======================================
    // EXTRA SAFETY / CORRECTION PATTERNS
    // ======================================

    // 11100 -> strong right
    else if (s0 && s1 && s2 && !s3 && !s4)
    {
        strongRight();
    }

    // 00011 -> strong left
    else if (!s0 && !s1 && !s2 && s3 && s4)
    {
        strongLeft();
    }

    // 11110 -> strong right
    else if (s0 && s1 && s2 && s3 && !s4)
    {
        strongRight();
    }

    // 01111 -> strong left
    else if (!s0 && s1 && s2 && s3 && s4)
    {
        strongLeft();
    }

    // 00000 = line lost
    else if (!s0 && !s1 && !s2 && !s3 && !s4)
    {
        stopMotors();
    }

    // 11111 = intersection / all sensors active
    else if (s0 && s1 && s2 && s3 && s4)
    {
        forward();
    }

    // Unknown pattern
    else
    {
        forward();
    }

    delay(5);
}
