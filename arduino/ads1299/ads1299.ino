/***************************************************************
  Instituto de Ingeniería Biomédica (IIBM)
  Facultad de Ingenieria, Universidad de Buenos Aires (FIUBA)

  Code description:
  This code controls the IC ADS1299 through an Arduino NANO.
  It handles the startup sequence (configuration, setup) of
  the device and triggers the data conversion.
  The processing loop reads data continuously, processes it
  and sends it to a computer through SPI.

  Florencia Grosso

  Note: Use old bootloader for Arduino Nano (ATmega328P) to
        avoid problems with SPI communication.

****************************************************************/

// Libraries
#include <SPI.h>
#include <stdio.h>
#include <string.h>

// ADS1299 master clock: 2 MHz, 500 ns per tCLK.
const uint16_t CS_HOLD_US = 2; // Rounded up from 4 tCLK.
const uint16_t RESET_WAIT_US = 9; // 18 tCLK at 2 MHz.
const float VREF_MV = 4500.0f;

// Commands
// System commands
#define WAKEUP 0x02
#define STANDBY 0x04
#define RESET 0x06
#define START 0x08
#define STOP 0x0A

// Data Read Commands
// Read Data Continuous mode
#define RDATAC 0x10
// Stop Read Data Continuously mode
#define SDATAC 0x11
// Read data by command.
#define RDATA 0x12

// Register Read Commands
#define RREG 0x20
#define WREG 0x40

// Registers
#define ID 0x00
#define CONFIG1 0x01
#define CONFIG2 0x02
#define CONFIG3 0x03
#define LOFF 0x04
#define CH1SET 0x05
#define CH2SET 0x06
#define CH3SET 0x07
#define CH4SET 0x08
#define CH5SET 0x09
#define CH6SET 0x0A
#define CH7SET 0x0B
#define CH8SET 0x0C
#define BIAS_SENSP 0x0D
#define BIAS_SENSN 0x0E
#define LOFF_SENSP 0x0F
#define LOFF_SENSN 0x10
#define LOFF_FLIP 0x11
#define LOFF_STATP 0x12
#define LOFF_STATN 0x13
#define GPIO 0x14
#define MISC1 0x15
#define MISC2 0x16
#define CONFIG4 0x17

// Pins for the arduino board
const int PIN_DRDY = 2;
const int PIN_START = 5;
const int PIN_SS = 10;
const int PIN_MOSI = 11; // DIN
const int PIN_MISO = 12; // DOUT
const int PIN_SCK = 13;
const int PIN_RESET = 4;

// Prototypes of used functions
void wake_up();
void send_command(uint8_t command);
void send_command_and_wait(uint8_t command, uint16_t time_to_wait_us);
byte fRREG(byte address);
void fWREG(byte address, byte data);
void get_device_id();

// Global variables
// Flag that indicates whether debug logs should be published
boolean verbose = false;
// Device id published by the ADS1299
byte device_id = 0;
// Number of active EEG channels
unsigned int active_channels = 2;
unsigned int configured_sample_rate = 250;
unsigned int configured_gain = 24;

bool is_supported_sample_rate(unsigned int sample_rate) {
  return sample_rate == 250 || sample_rate == 500 || sample_rate == 1000;
}

bool is_supported_gain(unsigned int gain) {
  return gain == 1 || gain == 2 || gain == 4 || gain == 6 ||
         gain == 8 || gain == 12 || gain == 24;
}

byte sample_rate_register_value(unsigned int sample_rate) {
  switch (sample_rate) {
    case 250: return 0x96;
    case 500: return 0x95;
    case 1000: return 0x94;
    default: return 0x96;
  }
}

byte gain_register_value(unsigned int gain) {
  switch (gain) {
    case 1: return 0x00;
    case 2: return 0x10;
    case 4: return 0x20;
    case 6: return 0x30;
    case 8: return 0x40;
    case 12: return 0x50;
    case 24: return 0x60;
    default: return 0x60;
  }
}

void start_acquisition() {
  digitalWrite(PIN_START, LOW);
  delay(150);
  send_command(START);
  delay(150);
  send_command(RDATAC);
}

void wait_for_configuration() {
  char command[32];

  while (true) {
    while (Serial.available() == 0) {
      delay(1);
    }

    char first = Serial.read();
    if (first == '1') {
      configured_sample_rate = 250;
      configured_gain = 24;
      start_acquisition();
      return;
    }
    if (first != 'C') {
      continue;
    }

    command[0] = first;
    byte length = 1;
    unsigned long started = millis();
    while (length < sizeof(command) - 1 && millis() - started < 100) {
      if (Serial.available() > 0) {
        char next = Serial.read();
        if (next == '\n') {
          break;
        }
        if (next != '\r') {
          command[length++] = next;
        }
      }
    }
    command[length] = '\0';

    unsigned int requested_rate = 0;
    unsigned int requested_gain = 0;
    if (strncmp(command, "CFG,", 4) != 0 ||
        sscanf(command + 4, "%u,%u", &requested_rate, &requested_gain) != 2 ||
        !is_supported_sample_rate(requested_rate) ||
        !is_supported_gain(requested_gain)) {
      Serial.println("ERROR,unsupported sample rate or gain");
      continue;
    }

    configured_sample_rate = requested_rate;
    configured_gain = requested_gain;
    send_command(SDATAC);
    fWREG(CONFIG1, sample_rate_register_value(configured_sample_rate));
    byte channel_setting = gain_register_value(configured_gain);
    fWREG(CH1SET, channel_setting);
    fWREG(CH2SET, channel_setting);
    Serial.print("READY,");
    Serial.print(configured_sample_rate);
    Serial.print(",");
    Serial.println(configured_gain);
    Serial.flush();
    start_acquisition();
    return;
  }
}

void setup() {
  /* Initialization routine for SPI bus */
  // Initalize communication with computer.
  // Open serial port (USB). 500000 baud is exact on a 16 MHz AVR.
  Serial.begin(500000, SERIAL_8N1);
  // Wait until data transmission through serial port is completed (in case there's a transmission going on)
  Serial.flush();
  delay(2000);

  // Initialize SPI before applying the ADS1299's mode 1 clock settings.
  SPI.begin();
  SPI.beginTransaction(SPISettings(2000000, MSBFIRST, SPI_MODE1));

  // Set Pin modes
  pinMode(PIN_DRDY, INPUT);
  pinMode(PIN_MISO, INPUT);
  pinMode(PIN_START, OUTPUT);
  pinMode(PIN_SS, OUTPUT);
  pinMode(PIN_MOSI, OUTPUT);
  pinMode(PIN_SCK, OUTPUT);

  // Enable device to configure it
  digitalWrite(PIN_START, LOW);
  digitalWrite(PIN_SS, HIGH);

  /* Follow initial flow at power-up (page 62) */
  // Delay>tPOR in ms (make sure thatVCAP1 1.1v = ok)
  delay(300);
  // Reset the device
  send_command_and_wait(RESET, RESET_WAIT_US);
  delay(500);
  send_command_and_wait(SDATAC, CS_HOLD_US);
  delay(300);

  /* TEST 1 */
  // Read ADS1299 ID to check that it receives data and there's a communication established
  while (device_id == 0) {
    get_device_id();
  }
  // Enable the internal reference and BIAS buffer with the internal BIASREF.
  fWREG(CONFIG3, 0xEC);
  fWREG(BIAS_SENSP, 0x03); // Bias IN1P + IN2P
  fWREG(BIAS_SENSN, 0x03); // Bias IN1N + IN2N

  // Define the Data rate (p.46)
  // [Set at minimum always, then changed by the desired value]
  fWREG(CONFIG1, sample_rate_register_value(configured_sample_rate));

  // Keep the internal test signal disabled for normal electrode acquisition.
  fWREG(CONFIG2, 0xC0);

  // Activate this for reference mode
  // fWREG(MISC1, 0x20);

  /* TEST 2 */
  // Read the written registers and print data
  if (verbose) {
    Serial.println("Check config regs:");
    Serial.print("CONFIG1: ");
    Serial.println(fRREG(CONFIG1), HEX);
    Serial.print("CONFIG2: ");
    Serial.println(fRREG(CONFIG2), HEX);
    Serial.print("CONFIG3: ");
    Serial.println(fRREG(CONFIG3), HEX);
  }

  // Configure all channels (p.50)
  //0x00 normal operation, unity gain & normal electrode input
  //0x60 normal operation, gain x24 & normal electrode input
  //0x01 normal operation, unity gain & input shorted
  //0x81 power-down, unity gain & input shorted
  //0x05 normal operation, unity gain & test signal
  fWREG(CH1SET, gain_register_value(configured_gain));
  fWREG(CH2SET, gain_register_value(configured_gain));
  fWREG(CH3SET, 0x81);
  fWREG(CH4SET, 0x81);
  fWREG(CH5SET, 0x81);
  fWREG(CH6SET, 0x81);
  fWREG(CH7SET, 0x81);
  fWREG(CH8SET, 0x81);

  /* TEST 3 */
  // Read configured channels and registers. Print data.
  if (verbose) {
    Serial.println("Check channels:");
    Serial.print("CH1SET: ");
    Serial.println(fRREG(CH1SET), HEX);
    Serial.print("CH4SET: ");
    Serial.println(fRREG(CH4SET), HEX);// we may have trouble here since it has to print a binary, if not use print(xxx, BIN)
  }

  // Wait for a host-supplied configuration before starting data transfer.
  // A single '1' keeps compatibility with the legacy MATLAB controller.
  wait_for_configuration();
}

// Operation loop for the Arduino
void loop() {
  // Wait until there is data available
  while (digitalRead(PIN_DRDY) == HIGH);
  if (verbose) {
    Serial.print("There is Data Ready.\n");
  }
  // Discard the 24-bit status word.
  digitalWrite(SS, LOW);
  SPI.transfer(0x00);
  SPI.transfer(0x00);
  SPI.transfer(0x00);

  // Now read the active channels
  for (int i=0; i < active_channels; i++){
    // Each channel has 24 bits
    byte c1 = SPI.transfer(0x00);
    byte c2 = SPI.transfer(0x00);
    byte c3 = SPI.transfer(0x00);

    uint32_t raw_code = ((uint32_t)c1 << 16) | ((uint32_t)c2 << 8) | c3;
    int32_t signed_code = (int32_t)(raw_code & 0x7FFFFFUL) -
                          (int32_t)(raw_code & 0x800000UL);
    float millivolts =
        ((float)signed_code * (2.0f * VREF_MV)) /
        ((float)configured_gain * 16777216.0f);

    // Send the input-referred differential voltage in millivolts.
    byte * data = (byte *)&millivolts;
    Serial.write(data,4);
  }
  delayMicroseconds(CS_HOLD_US);
  digitalWrite(SS, HIGH);
}

/* SYSTEM COMMANDS */

/* Wake-up from standby mode
 * TODO: Send this as a command too (once figured out how to handle
 * the intermediate delay).
 */
void wake_up() {
  // Set SS low to communicate with device.
  digitalWrite(SS, LOW);
  SPI.transfer(WAKEUP);
  delayMicroseconds(CS_HOLD_US);
  digitalWrite(SS, HIGH);
  delayMicroseconds(CS_HOLD_US);
}

/* Sends a command to the ADS1299.
 * Includes the required CS hold and command turnaround timing.
 * @param command: command to issue.
*/
void send_command(uint8_t command) {
  digitalWrite(SS, LOW);
  SPI.transfer(command);
  delayMicroseconds(CS_HOLD_US);
  digitalWrite(SS, HIGH);
  delayMicroseconds(CS_HOLD_US);
}

/* Sends a command to the ADS 1299.
 * Waits for the command execution interval afterwards.
 * @param command: command to issue.
 * @param time_to_wait: delay after issuing the command.
*/
void send_command_and_wait(uint8_t command, uint16_t time_to_wait_us) {
  digitalWrite(SS, LOW);
  SPI.transfer(command);
  delayMicroseconds(CS_HOLD_US);
  digitalWrite(SS, HIGH);
  delayMicroseconds(time_to_wait_us);
}

/* REGISTER READ/WRITE COMMANDS */

/* Read from register.
 * @param address: the starting register address.
 * @return the block of data read.
*/
byte fRREG(byte address) {
  // RREG expects 001rrrrr where rrrrr = _address
  byte op_code = RREG + address;
  digitalWrite(SS, LOW);
  SPI.transfer(op_code);
  SPI.transfer(0x00);
  byte data = SPI.transfer(0x00);
  delayMicroseconds(CS_HOLD_US);
  digitalWrite(SS, HIGH);
  delay(1);
  return data;
}

/* Writes to register.
 * @param address: the starting register address.
 * @param data: value to write.
*/
void fWREG(byte address, byte data) {
  // WREG expects 001rrrrr where rrrrr = _address
  // Open SPI.
  byte op_code = WREG + address;
  digitalWrite(SS, LOW);
  SPI.transfer(op_code);
  // Write only one register
  SPI.transfer(0x00);
  SPI.transfer(data);
  delayMicroseconds(CS_HOLD_US);
  digitalWrite(SS, HIGH);
  if (verbose) {
    Serial.print("Register 0x");
    Serial.print(address, HEX);
    Serial.println(" modified.\n");
  }
  delay(1);
}

/* Retrieves the device ID and stores it to device_id.
 */
void get_device_id() {
  byte data = fRREG(ID);
  // If retrieved ID is valid, then no power up is needed
  device_id = data;
  if (verbose) {
    Serial.println("Device ID: ");
    Serial.println(device_id, BIN);
  }
}
