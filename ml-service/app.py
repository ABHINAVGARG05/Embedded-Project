import numpy as np
import torch
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from model import FallCNN
from preprocessing import CHANNELS, WINDOW_SIZE

MODEL_PATH  = "model.pth"
SCALER_PATH = "scaler.pkl"
DEVICE      = torch.device("cpu")
THRESHOLD   = 0.5

model = FallCNN(in_channels=CHANNELS, window_size=WINDOW_SIZE).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

scaler = joblib.load(SCALER_PATH)

app = FastAPI(title="Fall Detection API", version="1.0.0")

class SensorWindow(BaseModel):
    acc: List[List[float]]

@app.post("/predict")
def predict(data: SensorWindow):
    acc = np.array(data.acc, dtype=np.float32)   

    if acc.shape[1] != 3:
        raise HTTPException(status_code=422, detail="acc must have 3 columns (x, y, z)")
    if acc.shape[0] != WINDOW_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {WINDOW_SIZE} time steps, got {acc.shape[0]}"
        )

    acc_norm = scaler.transform(acc)          

    tensor = torch.tensor(acc_norm.T[np.newaxis], dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        prob = model(tensor).item()

    return {
        "fall":       prob >= THRESHOLD,
        "confidence": round(prob, 4),
    }


@app.get("/health")
def health():
    return {"status": "ok"}