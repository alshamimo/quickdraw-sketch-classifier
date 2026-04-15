"""
FastAPI backend for drawing predictions.

Provides REST endpoints for processing canvas drawings and returning
class probabilities from both NN and CNN models.

Preprocessing pipeline converts raw canvas images to the same format
used during training (28x28 normalized grayscale).
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn.functional as F
import numpy as np
import base64
from PIL import Image
import io
import sys
import os

# Project imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.models import QuickDrawNN, QuickDrawCNN
import config

# ─── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="QuickDraw AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

# ─── Load Models ────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_class, path):
    model = model_class().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model

nn_model  = load_model(QuickDrawNN,  "train_results/nn_model.pth")
cnn_model = load_model(QuickDrawCNN, "train_results/cnn_model.pth")

print(f"Models loaded on: {DEVICE}")

# ─── Request Schema ───────────────────────────────────────────────────────────
class DrawingRequest(BaseModel):
    image: str  # Base64 PNG from canvas

# ─── Preprocessing ────────────────────────────────────────────────────────────
def preprocess_image(base64_string: str) -> torch.Tensor:
    """
    Convert base64 PNG from canvas to normalized tensor.

    Pipeline steps:
    1. Decode base64 and load PNG
    2. Convert to grayscale
    3. Apply threshold to remove light noise
    4. Find bounding box and center the drawing
    5. Resize to 28x28 (model input size)
    6. Normalize pixel values to [0, 1]

    Args:
        base64_string: Base64-encoded PNG image from frontend canvas

    Returns:
        Normalized tensor of shape (1, 1, 28, 28) ready for model input
    """
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]

    image_bytes = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("L")

    # Threshold: #0a0a0a background → pure black
    arr_raw = np.array(image)
    arr_raw[arr_raw < 30] = 0
    image = Image.fromarray(arr_raw)

    # Find bounding box + center
    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)
        w, h = image.size
        max_side = max(w, h)
        padding = int(max_side * 0.25)
        new_size = max_side + padding * 2
        padded = Image.new("L", (new_size, new_size), 0)
        offset_x = (new_size - w) // 2
        offset_y = (new_size - h) // 2
        padded.paste(image, (offset_x, offset_y))
        image = padded

    # Scale to 28x28 + normalize
    image = image.resize((28, 28), Image.LANCZOS)
    image.save("debug_input.png")
    arr = np.array(image).astype("float32") / 255.0
    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0).to(DEVICE)

    return tensor

# ─── Prediction ───────────────────────────────────────────────────────────────
def predict(model, tensor) -> dict:
    """
    Run inference and return class probabilities.

    Applies softmax to model logits to obtain normalized probabilities
    for each class. Results are rounded for cleaner API responses.

    Args:
        model: Neural network model (NN or CNN)
        tensor: Preprocessed input tensor

    Returns:
        Dictionary mapping class names to probability scores (0-1)
    """
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)
        probs  = probs.cpu().numpy()[0]

    return {
        cls: round(float(prob), 4)
        for cls, prob in zip(config.CLASSES, probs)
    }

# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.post("/predict")
async def predict_drawing(request: DrawingRequest):
    try:
        tensor    = preprocess_image(request.image)
        cnn_preds = predict(cnn_model, tensor)
        nn_preds  = predict(nn_model,  tensor)
        return {"cnn": cnn_preds, "nn": nn_preds}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "classes": config.CLASSES}
