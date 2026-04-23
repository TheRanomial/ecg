from __future__ import annotations

import base64
import io
import os
from typing import Dict, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from PIL import Image
from tensorflow.keras.models import load_model

CLASSES = ["HB", "MI", "PMI", "Normal"]
TARGET_SIZE = (224, 224)


class Base64PredictRequest(BaseModel):
    imageBase64: str


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    probabilities: Dict[str, float]


def parse_data_url_or_raw_base64(value: str) -> bytes:
    if value.startswith("data:"):
        parts = value.split(",", 1)
        if len(parts) != 2:
            raise ValueError("Invalid data URL.")
        value = parts[1]
    return base64.b64decode(value)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:  # pragma: no cover
        raise ValueError("Unsupported or invalid image.") from exc

    img = img.resize(TARGET_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def make_prediction(model, arr: np.ndarray) -> PredictionResponse:
    pred = model.predict(arr, verbose=0)[0]
    best_idx = int(np.argmax(pred))
    best_confidence = float(pred[best_idx]) * 100.0

    probabilities = {
        CLASSES[i]: round(float(pred[i]) * 100.0, 2) for i in range(len(CLASSES))
    }
    return PredictionResponse(
        label=CLASSES[best_idx],
        confidence=round(best_confidence, 2),
        probabilities=probabilities,
    )


def create_app() -> FastAPI:
    app = FastAPI(title="ECG Inference API", version="1.0.0")

    model_path = os.getenv("ECG_MODEL_PATH", "./model/resnet_ecg.h5")
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file not found: {model_path}")

    model = load_model(model_path)

    @app.get("/health")
    def health() -> Dict[str, object]:
        return {"ok": True, "modelLoaded": True, "classes": CLASSES}

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(image: UploadFile = File(...)) -> PredictionResponse:
        try:
            image_bytes = await image.read()
            if not image_bytes:
                raise ValueError("Empty image file.")
            arr = preprocess_image(image_bytes)
            return make_prediction(model, arr)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    @app.post("/predict-base64", response_model=PredictionResponse)
    async def predict_base64(payload: Base64PredictRequest) -> PredictionResponse:
        try:
            image_bytes = parse_data_url_or_raw_base64(payload.imageBase64)
            arr = preprocess_image(image_bytes)
            return make_prediction(model, arr)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return app


app = create_app()
