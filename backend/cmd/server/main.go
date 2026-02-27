package main

import (
	"log"

	"github.com/ABHINAVGARG05/embedded-project/internal/mqtt"
)

func main() {
	err := mqtt.StartSubscriber()
	if err != nil {
		log.Fatal(err)
	}

	select {}
}