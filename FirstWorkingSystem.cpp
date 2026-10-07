
#include <Arduino.h>

// [FIX-2] Optional AVR hardware watchdog. OFF by default, see the notes after the files.
#define USE_HW_WATCHDOG 0
#if USE_HW_WATCHDOG && defined(ARDUINO_ARCH_AVR)
#include <avr/wdt.h>
#endif

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

// [FIX-5] Debug output for diagnosis. Python prints these as "[ARDUINO] ...".
const bool DEBUG_SAFETY = true;
// Never print more often than this, so a flickering state can't slow the LFR loop.
const unsigned long DEBUG_MIN_INTERVAL_MS = 50;

// [FIX-5] "Line lost" beep pattern: 100 ms ON, 400 ms OFF (safety stop stays a SOLID tone).
const unsigned long LINE_LOST_BEEP_PERIOD_MS = 500;
const unsigned long LINE_LOST_BEEP_ON_MS = 100;

// ==========================================
// GLOBAL STATE
// ==========================================

bool obstacleDetected = true;
bool qrOutputs[10] = {false};
String serialBuffer = "";
unsigned long lastSafetyHeartbeatMs = 0;
unsigned long lastIRDebugMs = 0;

// [FIX-5] Set by lineFollowing() when all 5 IR sensors see no line.
bool lineLost = false;
unsigned long lastLineLostMs = 0;

// [FIX-5] Debug bookkeeping: what we last printed.
bool prevObstacleDebug = true;
bool prevLineLostDebug = false;
unsigned long lastDebugPrintMs = 0;

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

    // [FIX-5] Assume the line is found; only the "all sensors inactive" branch below sets it true.
    lineLost = false;

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
        lineLost = true;               // [FIX-5] remember that this stop is "line lost", not "safety"
        lastLineLostMs = millis();     // [FIX-5]
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
    // Safety stop or watchdog timeout: SOLID buzzer (unchanged behaviour).
    bool buzzerOn = obstacleDetected || timedOut;
    digitalWrite(BUZZER_PIN, buzzerOn ? HIGH : LOW);
}

// [FIX-5] Line lost (and NOT a safety stop): short beep, 100 ms on / 400 ms off.
// Runs after lineFollowing(), so it overrides the "buzzer off" that buzzerControl() just set.
void lineLostBeep()
{
    if (lineLost && !obstacleDetected)
    {
        bool beepOn = (millis() % LINE_LOST_BEEP_PERIOD_MS) < LINE_LOST_BEEP_ON_MS;
        digitalWrite(BUZZER_PIN, beepOn ? HIGH : LOW);
    }
}

// [FIX-5] Print only when something changed, and never faster than DEBUG_MIN_INTERVAL_MS.
// If a change can't be printed yet, it stays "changed" and is printed on a later pass.
void debugReport()
{
    if (!DEBUG_SAFETY)
    {
        return;
    }

    unsigned long now = millis();
    if (now - lastDebugPrintMs < DEBUG_MIN_INTERVAL_MS)
    {
        return;
    }

    if (obstacleDetected != prevObstacleDebug)
    {
        // hbAge < 2000  -> an explicit STOP/SAFE command caused it
        // hbAge >= 2000 -> the 2 s watchdog timed out (no heartbeats arriving)
        Serial.print("DBG obstacle=");
        Serial.print((int)obstacleDetected);
        Serial.print(" hbAge=");
        Serial.println(now - lastSafetyHeartbeatMs);
        prevObstacleDebug = obstacleDetected;
        lastDebugPrintMs = now;
    }
    else if (lineLost != prevLineLostDebug)
    {
        Serial.print("DBG lineLost=");
        Serial.println((int)lineLost);
        prevLineLostDebug = lineLost;
        lastDebugPrintMs = now;
    }
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
                // QR commands never touch lastSafetyHeartbeatMs.
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
#if USE_HW_WATCHDOG && defined(ARDUINO_ARCH_AVR)
    MCUSR = 0;          // [FIX-2] clear reset flags
    wdt_disable();      // [FIX-2] make sure an old watchdog setting can't fire during setup
#endif

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

#if USE_HW_WATCHDOG && defined(ARDUINO_ARCH_AVR)
    wdt_enable(WDTO_2S);   // [FIX-2] if loop() ever hangs, the chip resets and boots with obstacleDetected = true
#endif
}

// ==========================================
// MAIN LOOP
// ==========================================

void loop()
{
#if USE_HW_WATCHDOG && defined(ARDUINO_ARCH_AVR)
    wdt_reset();           // [FIX-2] "I'm still alive"
#endif

    processSerialCommands();
    buzzerControl();

    if (obstacleDetected)
    {
        stopMotors();
        lineLost = false;  // [FIX-5] a safety stop is not a "line lost" situation
    }
    else
    {
        lineFollowing();
    }

    lineLostBeep();        // [FIX-5]
    debugReport();         // [FIX-5]
    updateQROutputs();
}
