package mqtt

import (
	"fmt"
	"log"
	"os"

	"github.com/ABHINAVGARG05/embedded-project/internal/handler"
	mqtt "github.com/eclipse/paho.mqtt.golang"
)

func StartSubscriber() error {
	broker := os.Getenv("MQTT_BROKER")
	if broker == "" {
		broker = "tcp://localhost:1883"
	}

	opts := mqtt.NewClientOptions()
	opts.AddBroker(broker)
	opts.SetClientID("fwds-backend")

	opts.SetDefaultPublishHandler(messageHandler)

	client := mqtt.NewClient(opts)
	if token := client.Connect(); token.Wait() && token.Error() != nil {
		return token.Error()
	}

	if token := client.Subscribe("fwds/device/+/+", 1, nil); token.Wait() && token.Error() != nil {
		return token.Error()
	}

	fmt.Println("MQTT Subscriber started...")
	return nil
}

func messageHandler(client mqtt.Client, msg mqtt.Message) {
	log.Printf("Received on topic %s: %s\n", msg.Topic(), msg.Payload())
	handler.ProcessMessage(msg.Topic(), msg.Payload())
}