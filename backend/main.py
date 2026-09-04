import os
import shutil
import numpy as np

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    BatchNormalization
)

import tensorflow as tf
import keras

print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)

# ==============================
# BUILD MODEL ARCHITECTURE
# ==============================

base_model = MobileNetV2(
    weights=None,
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(6, activation="softmax")
])

# ==============================
# LOAD MODEL WEIGHTS
# ==============================

model.load_weights(
    "model/steel_weights.weights.h5"
)

print("Model Weights Loaded Successfully")

# ==============================
# CLASS LABELS
# ==============================

class_names = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches"
]

# ==============================
# IMAGE SETTINGS
# ==============================

IMG_SIZE = 224

# ==============================
# CREATE FASTAPI APP
# ==============================

app = FastAPI(
    title="Steel Surface Defect Analysis API",
    description=(
        "Backend service for the Defect Analysis System. Accepts a steel "
        "surface image and returns the predicted defect category with a "
        "confidence score."
    ),
    version="1.0.0",
)

# ==============================
# ENABLE CORS
# ==============================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# CREATE UPLOADS FOLDER
# ==============================

os.makedirs("uploads", exist_ok=True)

# ==============================
# HOME ROUTE
# ==============================

@app.get("/")
def home():
    return {
        "status": "ok",
        "service": "Steel Surface Defect Analysis API",
        "defect_classes": class_names,
    }


@app.get("/health")
def health():
    return {"status": "ok"}

# ==============================
# PREDICTION FUNCTION
# ==============================

def predict_defect(img_path):

    img = image.load_img(
        img_path,
        target_size=(IMG_SIZE, IMG_SIZE)
    )

    img_array = image.img_to_array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    prediction = model.predict(
        img_array,
        verbose=0
    )

    predicted_class = class_names[
        np.argmax(prediction)
    ]

    confidence = float(
        np.max(prediction) * 100
    )

    return {
        "prediction": predicted_class,
        "confidence": round(confidence, 2)
    }

# ==============================
# PREDICTION API
# ==============================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    file_path = f"uploads/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer
        )

    result = predict_defect(file_path)

    return result