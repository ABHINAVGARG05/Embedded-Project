package main

import (
	"log"
	"net/http"

	"github.com/ABHINAVGARG05/embedded-project/internal/handler"
)

func main() {

	http.HandleFunc("/api/sensor", handler.SensorHandler)

	log.Println("FWDS Backend running on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}