package handler

import (
	"encoding/json"
	"fmt"
	"strings"
)

type FallAlert struct {
	DeviceID string  `json:"device_id"`
	Fall     bool    `json:"fall"`
	Lat      float64 `json:"lat"`
	Lon      float64 `json:"lon"`
	Battery  int     `json:"battery"`
}

func ProcessMessage(topic string, payload []byte) {
	parts := strings.Split(topic, "/")
	deviceID := parts[2]

	var alert FallAlert
	err := json.Unmarshal(payload, &alert)
	if err != nil {
		fmt.Println("Invalid payload:", err)
		return
	}

	alert.DeviceID = deviceID

	fmt.Printf("Fall detected from device %s at location (%f, %f)\n",
		alert.DeviceID, alert.Lat, alert.Lon)

	// TODO:
	// - Store in DB
	// - Trigger push notification
}