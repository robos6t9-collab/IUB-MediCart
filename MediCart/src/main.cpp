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
// BUZZER AND QR OUTPUT PINS
// ==========================================

const int BUZZER_PIN = 24;

// 10 available digital output pins that do not conflict with:
// - motors 2,3,4,5
// - IR sensors A8-A12
// - buzzer pin 24
// - serial pins 0,1
const int QR_OUTPUT_PINS[10] = {6, 7, 8, 9, 10, 11, 12, 13, 22, 23};

// QR output mapping:
// QR 1 -> pin 6
// QR 2 -> pin 7
// QR 3 -> pin 8
// QR 4 -> pin 9
// QR 5 -> pin 10
// QR 6 -> pin 11
// QR 7 -> pin 12
// QR 8 -> pin 13
// QR 9 -> pin 22
// QR 10 -> pin 23

// ==========================================
// SETTINGS
// ==========================================

// true  = black line on white surface
// false = white line on black surface
const bool BLACK_LINE = true;

// Motor speeds
const int FORWARD_SPEED = 150;
const int SLIGHT_SPEED = 120;
const int STRONG_SPEED = 180;

const unsigned long COMMAND_TIMEOUT_MS = 2000;
const bool DEBUG_IR_SENSORS = false;

// ==========================================
// GLOBAL STATE
// ==========================================

bool obstacleDetected = true;
bool qrOutputs[10] = {false};
String serialBuffer = "";
unsigned long lastSafetyHeartbeatMs = 0;
unsigned long lastIRDebugMs = 0;

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

    if (BLACK_LINE)
    {
        return value == LOW;
    }
    else
    {
        return value == HIGH;
    }
}

void readIRSensors(bool &s0, bool &s1, bool &s2, bool &s3, bool &s4)
{
    s0 = sensorActive(0);
    s1 = sensorActive(1);
    s2 = sensorActive(2);
    s3 = sensorActive(3);
    s4 = sensorActive(4);
}

// ==========================================
// LINE FOLLOWING
// ==========================================

void lineFollowing()
{
    if (obstacleDetected)
    {
        stopMotors();
        return;
    }

    bool s0, s1, s2, s3, s4;
    readIRSensors(s0, s1, s2, s3, s4);

    if (DEBUG_IR_SENSORS && millis() - lastIRDebugMs >= 1000)
    {
        Serial.print(s0);
        Serial.print(s1);
        Serial.print(s2);
        Serial.print(s3);
        Serial.println(s4);
        lastIRDebugMs = millis();
    }

    // Existing logic preserved
    if (!s0 && !s1 && s2 && !s3 && !s4)
    {
        forward();
    }
    else if (!s0 && s1 && s2 && !s3 && !s4)
    {
        slightRight();
    }
    else if (s0 && s1 && !s2 && !s3 && !s4)
    {
        strongRight();
    }
    else if (!s0 && !s1 && s2 && s3 && !s4)
    {
        slightLeft();
    }
    else if (!s0 && !s1 && !s2 && s3 && s4)
    {
        strongLeft();
    }
    else if (s0 && s1 && s2 && !s3 && !s4)
    {
        strongRight();
    }
    else if (!s0 && s1 && s2 && s3 && s4)
    {
        strongLeft();
    }
    else if (s0 && s1 && s2 && s3 && !s4)
    {
        strongRight();
    }
    else if (!s0 && !s1 && !s2 && !s3 && !s4)
    {
        stopMotors();
    }
    else if (s0 && s1 && s2 && s3 && s4)
    {
        forward();
    }
    else
    {
        forward();
    }
}

// ==========================================
// BUZZER AND QR OUTPUT CONTROL
// ==========================================

void buzzerControl()
{
    unsigned long now = millis();
    bool timedOut = (now - lastSafetyHeartbeatMs) >= COMMAND_TIMEOUT_MS;
    if (timedOut)
    {
        obstacleDetected = true;
    }
    bool buzzerOn = obstacleDetected || timedOut;
    digitalWrite(BUZZER_PIN, buzzerOn ? HIGH : LOW);
}

void updateQROutputs()
{
    for (int i = 0; i < 10; i++)
    {
        digitalWrite(QR_OUTPUT_PINS[i], qrOutputs[i] ? HIGH : LOW);
    }
}

void resetQROutputs()
{
    for (int i = 0; i < 10; i++)
    {
        qrOutputs[i] = false;
    }
}

// ==========================================
// SERIAL COMMAND PARSING
// ==========================================

void processSerialCommands()
{
    while (Serial.available() > 0)
    {
        char incoming = Serial.read();

        if (incoming == '\n' || incoming == '\r')
        {
            String command = serialBuffer;
            serialBuffer = "";
            command.trim();

            if (command.length() == 0)
            {
                continue;
            }

            if (command == "STOP")
            {
                lastSafetyHeartbeatMs = millis();
                obstacleDetected = true;
            }
            else if (command == "SAFE")
            {
                lastSafetyHeartbeatMs = millis();
                obstacleDetected = false;
            }
            else if (command == "RESET_QR")
            {
                resetQROutputs();
            }
            else if (command.startsWith("QR:"))
            {
                int qrValue = command.substring(3).toInt();
                if (qrValue >= 1 && qrValue <= 10)
                {
                    resetQROutputs();
                    qrOutputs[qrValue - 1] = true;
                }
            }
        }
        else
        {
            serialBuffer += incoming;
        }
    }

    if ((millis() - lastSafetyHeartbeatMs) >= COMMAND_TIMEOUT_MS)
    {
        obstacleDetected = true;
        stopMotors();
    }
}

// ==========================================
// SETUP
// ==========================================

void setup()
{
    Serial.begin(115200);
    serialBuffer.reserve(32);

    for (int i = 0; i < 5; i++)
    {
        pinMode(IR_PINS[i], INPUT);
    }

    pinMode(MOTOR_L_PIN1, OUTPUT);
    pinMode(MOTOR_L_PIN2, OUTPUT);
    pinMode(MOTOR_R_PIN1, OUTPUT);
    pinMode(MOTOR_R_PIN2, OUTPUT);

    pinMode(BUZZER_PIN, OUTPUT);

    for (int i = 0; i < 10; i++)
    {
        pinMode(QR_OUTPUT_PINS[i], OUTPUT);
        digitalWrite(QR_OUTPUT_PINS[i], LOW);
    }

    stopMotors();
    digitalWrite(BUZZER_PIN, LOW);
    lastSafetyHeartbeatMs = millis();

    Serial.println("MediCart ready");
}

// ==========================================
// MAIN LOOP
// ==========================================

void loop()
{
    processSerialCommands();
    buzzerControl();

    if (obstacleDetected)
    {
        stopMotors();
    }
    else
    {
        lineFollowing();
    }

    updateQROutputs();
}