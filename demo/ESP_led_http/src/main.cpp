#include <Arduino.h>
#include <ESP8266WiFi.h>
#include <WiFiClient.h>
#include <ESP8266WebServer.h>

const char* ssid = "TIM-30888077";
const char* password = "oBI0YuB1EaR5J2HEXgyQhLH9";

ESP8266WebServer server(80);

#define LED1 4
#define LED2 5
#define LED3 16

void led1();
void led2();
void led3();
void allon();
void alloff();
void magic();

bool led1State = false;
bool led2State = false;
bool led3State = false;

void setup() {
  Serial.begin(9600);

  Serial.print("\nConnecting to WiFi");
  WiFi.hostname("myledserver"); // server reachable e.g. at http://myledserver/led1
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.println("WiFi connected.");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());

  server.on("/led1", led1);
  server.on("/led2", led2);
  server.on("/led3", led3);
  server.on("/allon", allon);
  server.on("/alloff", alloff);
  server.on("/magic", magic);

  server.begin();
  Serial.println("Webserver started");

  pinMode(LED1, OUTPUT);
  pinMode(LED2, OUTPUT);
  pinMode(LED3, OUTPUT);
}

void loop() {
  server.handleClient();
}

void led1() {
  Serial.println("led1");
  led1State = !led1State;
  digitalWrite(LED1, led1State);
  server.send(200, "text/plain", "success");
}

void led2() {
  Serial.println("led2");
  led2State = !led2State;
  digitalWrite(LED2, led2State);
  server.send(200, "text/plain", "success");
}

void led3() {
  Serial.println("led3");
  led3State = !led3State;
  digitalWrite(LED3, led3State);
  server.send(200, "text/plain", "success");
}

void allon() {
  led1State = true;
  led2State = true;
  led3State = true;
  digitalWrite(LED1, led1State);
  digitalWrite(LED2, led2State);
  digitalWrite(LED3, led3State);
  server.send(200, "text/plain", "success");
}

void alloff() {
  led1State = false;
  led2State = false;
  led3State = false;
  digitalWrite(LED1, led1State);
  digitalWrite(LED2, led2State);
  digitalWrite(LED3, led3State);
  server.send(200, "text/plain", "success");
}

void onoffFromLeft() {
  led1State = false;
  led2State = false;
  led3State = false;
  digitalWrite(LED1, led1State);
  digitalWrite(LED2, led2State);
  digitalWrite(LED3, led3State);

  digitalWrite(LED1, true);
  delay(100);
  digitalWrite(LED1, false);

  digitalWrite(LED2, true);
  delay(100);
  digitalWrite(LED2, false);

  digitalWrite(LED3, true);
  delay(100);
  digitalWrite(LED3, false);
}

void magic() {
  Serial.println("magic");
  onoffFromLeft();
  onoffFromLeft();
  onoffFromLeft();

  led1State = true;
  led2State = true;
  led3State = true;
  digitalWrite(LED1, led1State);
  digitalWrite(LED2, led2State);
  digitalWrite(LED3, led3State);
  
  delay(200);

  led1State = false;
  led2State = false;
  led3State = false;
  digitalWrite(LED1, led1State);
  digitalWrite(LED2, led2State);
  digitalWrite(LED3, led3State);

  delay(200);

  led1State = true;
  led2State = true;
  led3State = true;
  digitalWrite(LED1, led1State);
  digitalWrite(LED2, led2State);
  digitalWrite(LED3, led3State);
  
  delay(200);

  led1State = false;
  led2State = false;
  led3State = false;
  digitalWrite(LED1, led1State);
  digitalWrite(LED2, led2State);
  digitalWrite(LED3, led3State);

  delay(200);

  led1State = true;
  led2State = true;
  led3State = true;
  digitalWrite(LED1, led1State);
  digitalWrite(LED2, led2State);
  digitalWrite(LED3, led3State);
  
  delay(200);

  led1State = false;
  led2State = false;
  led3State = false;
  digitalWrite(LED1, led1State);
  digitalWrite(LED2, led2State);
  digitalWrite(LED3, led3State);

  server.send(200, "text/plain", "success");
}