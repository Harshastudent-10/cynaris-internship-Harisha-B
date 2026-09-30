import os
import joblib
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

class IrisPredictionInput(BaseModel):
    features: List[float] = Field(
        ..., 
        json_schema_extra={"example": [5.1, 3.5, 1.4, 0.2]},
        description="List of 4 numeric features: sepal length, sepal width, petal length, petal width."
    )

class PredictionOutput(BaseModel):
    prediction: int
    class_name: str

CLASS_NAMES = ["setosa", "versicolor", "virginica"]
MODEL_PATH = "model_joblib.joblib"
model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"[INFO] Model loaded successfully from {MODEL_PATH}")
    else:
        print(f"[WARNING] Model file {MODEL_PATH} not found.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(
    title="FastAPI Model Serving Endpoint",
    description="API for serving ML predictions via FastAPI",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
def read_root():
    return {"status": "online", "message": "FastAPI Model Serving Endpoint is running."}

@app.post("/predict", response_model=PredictionOutput)
def predict(data: IrisPredictionInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    if len(data.features) != 4:
        raise HTTPException(status_code=400, detail="Expected 4 features.")

    input_array = np.array(data.features).reshape(1, -1)
    pred_class_idx = int(model.predict(input_array)[0])
    
    return PredictionOutput(
        prediction=pred_class_idx,
        class_name=CLASS_NAMES[pred_class_idx]
    )