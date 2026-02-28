package model

type SensorBatch struct {
	DeviceID string        `json:"device_id"`
	Acc      [][]float64   `json:"acc"`
}

type MLResponse struct {
	Fall bool `json:"fall"`
}